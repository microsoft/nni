# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
import tensorrt as trt
import numpy as np
import torch
from torch.functional import Tensor

from . import frontend_to_onnx as fonnx
from . import calibrator as calibrator
from . import trt_pycuda as common
from .backend import BaseModelSpeedup

# TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
TRT_LOGGER = trt.Logger()

class CalibrateType:
    LEGACY = trt.CalibrationAlgoType.LEGACY_CALIBRATION
    ENTROPY = trt.CalibrationAlgoType.ENTROPY_CALIBRATION
    ENTROPY2 = trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
    MINMAX = trt.CalibrationAlgoType.MINMAX_CALIBRATION

Precision_Dict = {
    8: trt.float32,
    16: trt.float16,
    32: trt.float32
}

def build_engine(model_file, calib, config=None, extra_layer_bit=32, strict_datatype=False):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, \
        trt.OnnxParser(network, TRT_LOGGER) as parser:
        """
        This function builds an engine from an onnx model with calibration process.
        """
        # Attention that, builder should be set to 1 because of the implementation of allocate_buffer
        builder.max_batch_size = 1
        builder.max_workspace_size = common.GiB(1)

        if extra_layer_bit == 32 and config is None:
            pass
        elif extra_layer_bit == 8 and config is None:
            # entire model in 8bit mode
            builder.int8_mode = True
            builder.int8_calibrator = calib
        else:
            builder.int8_mode = True
            builder.fp16_mode = True
            builder.int8_calibrator = calib
            builder.strict_type_constraints = strict_datatype

        # Parse onnx model
        with open(model_file, 'rb') as model:
            if not parser.parse(model.read()):
                print ('ERROR: Fail to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None

        # This design may not be correct if output more than one
        for i in range(network.num_layers):
            if config is None:
                break
            layer = network.get_layer(i)
            if layer.name in config:
                w_bit = config[layer.name]['weight_bit']
                a_bit = config[layer.name]['activation_bit']
                layer.precision = Precision_Dict[w_bit]
                layer.set_output_type(0, Precision_Dict[a_bit])
        # Build engine and do int8 calibration.
        engine = builder.build_cuda_engine(network)
        return engine

def build_engine_without_calib(model_file, config=None, extra_layer_bit=32, strict_datatype=False):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, \
        trt.OnnxParser(network, TRT_LOGGER) as parser:
        """
        This function builds an engine from an onnx model without calibration process. Users must provide config
        which contains quantization parameters for tensorrt to generate low bit inference engine.
        """
        # Attention that, builder should be set to 1 because of the implementation of allocate_buffer
        builder.max_batch_size = 1
        builder.max_workspace_size = common.GiB(1)

        if extra_layer_bit == 32 and config is None:
            pass
        elif extra_layer_bit == 16 and config is None:
            builder.fp16_mode = True
        elif extra_layer_bit == 8 and config is None:
            # entire model in 8bit mode
            builder.int8_mode = True
        else:
            pass
            builder.int8_mode = True
            builder.fp16_mode = True
            builder.strict_type_constraints = strict_datatype

        # Parse onnx model
        with open(model_file, 'rb') as model:
            if not parser.parse(model.read()):
                print ('ERROR: Fail to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None

        # This implementation may be incorrect when output number > 1
        for i in range(network.num_layers):
            if config is None:
                # no low bit layer need to be set, keep original model
                break
            layer = network.get_layer(i)
            if layer.name not in config:
                continue

            # If weight_bit exists in config, set layer precision and layer's input tensor dynamic range.
            # If layer's type is Gemm, do the same thing to layers before and after it.
            if 'weight_bit' in config[layer.name]:
                assert 'tracked_min_input' in config[layer.name]
                assert 'tracked_max_input' in config[layer.name]
                w_bit = config[layer.name]['weight_bit']
                tracked_min_input = config[layer.name]['tracked_min_input']
                tracked_max_input = config[layer.name]['tracked_max_input']
                layer.precision = Precision_Dict[w_bit]
                in_tensor = layer.get_input(0)
                in_tensor.dynamic_range = (tracked_min_input, tracked_max_input)
                # Gemm will generate two shuffle layers before and after itself, need specific setting
                if layer.name[0:4] == "Gemm":
                    pre_layer = network.get_layer(i-1)
                    pre_layer.precision = Precision_Dict[w_bit]
                    pre_in_tensor = pre_layer.get_input(0)
                    pre_in_tensor.dynamic_range = (tracked_min_input, tracked_max_input)
                    next_layer = network.get_layer(i+1)
                    next_layer.precision = Precision_Dict[w_bit]
                    next_in_tensor = next_layer.get_input(0)
                    next_in_tensor.dynamic_range = (tracked_min_input, tracked_max_input)

            # If activation exists in config, set layer output type and layer's output tensor dynamic range.
            # If layer's type is Gemm, do the same thing to layers before and after it.
            if 'activation_bit' in config[layer.name]:
                assert 'tracked_min_activation' in config[layer.name]
                assert 'tracked_max_activation' in config[layer.name]
                a_bit = config[layer.name]['activation_bit']
                tracked_min_activation = config[layer.name]['tracked_min_activation']
                tracked_max_activation = config[layer.name]['tracked_max_activation']
                layer.set_output_type(0, Precision_Dict[a_bit])
                out_tensor = layer.get_output(0)
                out_tensor.dynamic_range = (tracked_min_activation, tracked_max_activation)
                # Gemm will generate two shuffle layers before and after itself, need specific setting
                if layer.name[0:4] == "Gemm":
                    pre_layer = network.get_layer(i-1)
                    pre_layer.set_output_type(0, Precision_Dict[a_bit])
                    pre_out_tensor = pre_layer.get_output(0)
                    pre_out_tensor.dynamic_range = (tracked_min_activation, tracked_max_activation)
                    next_layer = network.get_layer(i+1)
                    next_layer.set_output_type(0, Precision_Dict[a_bit])
                    next_out_tensor = next_layer.get_output(0)
                    next_out_tensor.dynamic_range = (tracked_min_activation, tracked_max_activation)

        # Build engine and do int8 calibration.
        engine = builder.build_cuda_engine(network)
        return engine

class ModelSpeedupTensorRT(BaseModelSpeedup):
    def __init__(self, model, input_shape, config=None, onnx_path="default_model.onnx", extra_layer_bit=32, strict_datatype=True,
        calibrate_type=CalibrateType.ENTROPY2, calib_data_loader=None, calibration_cache = "calibration.cache", batchsize=1,
        input_names=["actual_input_1"], output_names=["output1"]):
        """
        Parameters
        ----------
        model : pytorch model
            The model to speed up by quantization.
        input_shape : tuple
            The input shape of model, shall pass it to torch.onnx.export.
        config : dict
            Config recording bit number and name of layers.
        onnx_path : str
            The path user want to store onnx model which is converted from pytorch model.
        extra_layer_bit : int
            Other layers which are not in config will be quantized to corresponding bit number.
        strict_datatype : bool
            Whether constrain layer bit to the number given in config or not. If true, all the layer
            will be set to given bit strictly. Otherwise, these layers will be set automatically by
            tensorrt.
        calibrate_type : tensorrt.tensorrt.CalibrationAlgoType
            The algorithm of calibrating. Please refer to https://docs.nvidia.com/deeplearning/
            tensorrt/api/python_api/infer/Int8/Calibrator.html for detail
        calibrate_data : numpy array
            The data using to calibrate quantization model
        calibration_cache : str
            The path user want to store calibrate cache file
        batchsize : int
            The batch size of calibration and inference
        input_names : list
            Input name of onnx model providing for torch.onnx.export to generate onnx model
        output_name : list
            Output name of onnx model providing for torch.onnx.export to generate onnx model
        """
        super().__init__(model, config)
        self.model = model
        self.onnx_path = onnx_path
        self.input_shape = input_shape
        self.config = config
        self.extra_layer_bit = extra_layer_bit
        self.strict_datatype = strict_datatype
        self.calibrate_type = calibrate_type
        self.calib_data_loader = calib_data_loader
        self.calibration_cache = calibration_cache
        self.batchsize = batchsize
        self.input_names = input_names
        self.output_names = output_names
        self.context = None
        self.onnx_config = {}

    def compress(self):
        """
        Get onnx config and build tensorrt engine.
        """
        assert self.model is not None
        assert self.onnx_path is not None
        assert self.input_shape is not None

        # Convert pytorch model to onnx model and save onnx model in onnx_path
        _, self.onnx_config = fonnx.torch_to_onnx(self.model, self.config, input_shape=self.input_shape,
            model_path=self.onnx_path, input_names=self.input_names, output_names=self.output_names)

        if self.calib_data_loader is not None:
            assert self.calibrate_type is not None
            context = self._tensorrt_build_withcalib(self.onnx_path)
        else:
            context = self._tensorrt_build_withoutcalib(self.onnx_path)
        self.context = context

    def _tensorrt_build_withcalib(self, onnx_path):
        # convert pytorch tensor to numpy darray
        calib_data = None
        if type(self.calib_data_loader) == torch.utils.data.dataloader.DataLoader:
            calib_data_set = []
            for data, _ in self.calib_data_loader:
                calib_data_set.append(data)
            calib_data = np.concatenate(calib_data_set)
        elif type(self.calib_data_loader) == torch.Tensor:
            calib_data = self.calib_data_loader.numpy()
        else:
            raise ValueError("Not support calibration datatype")
        calib = calibrator.Calibrator(calib_data, self.calibration_cache, self.batchsize, self.calibrate_type)

        # build inference engine with calibration
        engine = build_engine(onnx_path, calib, self.onnx_config, self.extra_layer_bit, self.strict_datatype)
        return engine.create_execution_context()

    def _tensorrt_build_withoutcalib(self, onnx_path):
        # build inference engine without calibration
        engine = build_engine_without_calib(onnx_path, self.onnx_config, self.extra_layer_bit, self.strict_datatype)
        return engine.create_execution_context()

    def inference(self, test_data):
        """
        Do inference by tensorrt builded engine.
        """
        # convert pytorch tensor to numpy darray
        test_data = test_data.numpy()
        # Numpy dtype should be float32
        assert test_data.dtype == np.float32
        elapsed_time = 0
        inputs, outputs, bindings, stream = common.allocate_buffers(self.context.engine)
        result = []
        for start_idx in range(0, test_data.shape[0], self.batchsize):
            # If the number of images in the test set is not divisible by the batch size, the last batch will be smaller.
            # This logic is used for handling that case.
            end_idx = min(start_idx + self.batchsize, test_data.shape[0])
            effective_batch_size = end_idx - start_idx

            # Do inference for every batch.
            inputs[0].host = test_data[start_idx:start_idx + effective_batch_size]
            t1 = time.time()
            [output] = common.do_inference_v2(self.context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            elapsed_time += time.time() - t1
            shape = output.shape[0]
            output = output[0:int(shape * effective_batch_size / self.batchsize)].reshape(effective_batch_size, -1)
            result.append(output.copy())
            # Use argmax to get predictions and then check accuracy
        # convert numpy darray to pytorch tensor
        result = torch.Tensor(np.concatenate(result))
        return result, elapsed_time

    def export_quantized_model(self, path):
        assert path is not None
        with open(path, "wb") as f:
            f.write(self.context.engine.serialize())
            print("TensorRT engine has been saved to", path)

    def load_quantized_model(self, path):
        assert path is not None
        with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            self.context = engine.create_execution_context()
            print("Load TensorRT engine from {} successfully".format(path))