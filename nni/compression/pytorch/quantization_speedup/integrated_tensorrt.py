# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
import logging
import tensorrt as trt
import numpy as np
import torch

from . import frontend_to_onnx as fonnx
from . import calibrator as calibrator
from . import trt_pycuda as common
from .backend import BaseModelSpeedup

TRT8 = 8
TRT7 = 7
TRT_LOGGER = trt.Logger()
logger = logging.getLogger(__name__)

class CalibrateType:
    LEGACY = trt.CalibrationAlgoType.LEGACY_CALIBRATION
    ENTROPY = trt.CalibrationAlgoType.ENTROPY_CALIBRATION
    ENTROPY2 = trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
    MINMAX = trt.CalibrationAlgoType.MINMAX_CALIBRATION

Precision_Dict = {
    8: trt.int8,
    16: trt.float16,
    32: trt.float32
}

def valid_config(config=None):
    """
    This function validates the bits setting configuration
    """
    if config is None:
        return
    support_bits = [8, 16, 32]
    for name in config.keys():
        if 'weight_bits' in config[name]:
            w_bits = config[name]['weight_bits']
            assert w_bits in support_bits, "weight bits should be 8, 16, 32"
        if 'output_bits' in config[name]:
            a_bits = config[name]['output_bits']
            assert a_bits in support_bits, "output bits should be 8, 16, 32"

def handle_gemm(network, layer_idx, config):
    """
    This function handles special gemm operation due to layer numbers of gemm changed during pytorch->onnx model convertion.

    Parameters
    ----------
    network : tensorrt.INetworkDefinition
        Represents a TensorRT Network from which the Builder can build an Engine
    layer_idx : int
        layer index of gemm
    config : dict
        Config recording bits number and name of layers
    """
    layer = network.get_layer(layer_idx)
    pre_layer = network.get_layer(layer_idx-1)
    next_layer = network.get_layer(layer_idx+1)
    # if weight bits exists, set three layers' precision,
    # input tensor range and the first two layers' output type
    if 'weight_bits' in config[layer.name]:
        assert 'tracked_min_input' in config[layer.name]
        assert 'tracked_max_input' in config[layer.name]
        w_bits = config[layer.name]['weight_bits']
        tracked_min_input = config[layer.name]['tracked_min_input']
        tracked_max_input = config[layer.name]['tracked_max_input']
        # set three layers the same precision
        layer.precision = Precision_Dict[w_bits]
        pre_layer.precision = Precision_Dict[w_bits]
        next_layer.precision = Precision_Dict[w_bits]
        # set the first two layers' output type
        pre_layer.set_output_type(0, Precision_Dict[w_bits])
        layer.set_output_type(0, Precision_Dict[w_bits])
        pre_in_tensor = pre_layer.get_input(0)
        in_tensor = layer.get_input(0)
        next_in_tensor = next_layer.get_input(0)
        # set three layers' input tensor range
        pre_in_tensor.dynamic_range = (tracked_min_input, tracked_max_input)
        in_tensor.dynamic_range = (tracked_min_input, tracked_max_input)
        next_in_tensor.dynamic_range = (tracked_min_input, tracked_max_input)

    # if output bits exists, set the last layer's output type output tensor range
    if 'output_bits' in config[layer.name]:
        assert 'tracked_min_output' in config[layer.name]
        assert 'tracked_max_output' in config[layer.name]
        a_bits = config[layer.name]['output_bits']
        tracked_min_output = config[layer.name]['tracked_min_output']
        tracked_max_output = config[layer.name]['tracked_max_output']
        # set the last layer's output type
        next_layer.set_output_type(0, Precision_Dict[a_bits])
        next_out_tensor = next_layer.get_output(0)
        # set the last layer's output tensor range
        next_out_tensor.dynamic_range = (tracked_min_output, tracked_max_output)

def build_engine(model_file, config=None, extra_layer_bits=32, strict_datatype=False, calib=None):
    """
    This function builds an engine from an onnx model with calibration process.

    Parameters
    ----------
    model_file : str
        The path of onnx model
    config : dict
        Config recording bits number and name of layers
    extra_layer_bits : int
        Other layers which are not in config will be quantized to corresponding bits number
    strict_datatype : bool
        Whether constrain layer bits to the number given in config or not. If true, all the layer
        will be set to given bits strictly. Otherwise, these layers will be set automatically by
        tensorrt
    calib : numpy array
        The data using to calibrate quantization model

    Returns
    -------
    tensorrt.ICudaEngine
        An ICudaEngine for executing inference on a built network
    """
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, \
        trt.OnnxParser(network, TRT_LOGGER) as parser, builder.create_builder_config() as trt_config:
        # Attention that, builder should be set to 1 because of the implementation of allocate_buffer
        trt_version = int(trt.__version__[0])
        assert trt_version == TRT8 or trt_version == TRT7, "Version of TensorRT is too old, please \
            update TensorRT to version >= 7.0"
        if trt_version == TRT7:
            logger.warning("TensorRT7 is deprecated and may be removed in the following release.")

        builder.max_batch_size = 1
        if trt_version == TRT8:
            trt_config.max_workspace_size = common.GiB(4)
        else:
            builder.max_workspace_size = common.GiB(4)

        if extra_layer_bits == 32 and config is None:
            pass
        elif extra_layer_bits == 16 and config is None:
            if trt_version == TRT8:
                trt_config.set_flag(trt.BuilderFlag.FP16)
            else:
                builder.fp16_mode = True
        elif extra_layer_bits == 8 and config is None:
            # entire model in 8bit mode
            if trt_version == TRT8:
                trt_config.set_flag(trt.BuilderFlag.INT8)
            else:
                builder.int8_mode = True
        else:
            if trt_version == TRT8:
                trt_config.set_flag(trt.BuilderFlag.INT8)
                trt_config.set_flag(trt.BuilderFlag.FP16)
                if strict_datatype:
                    trt_config.set_flag(trt.BuilderFlag.STRICT_TYPES)
            else:
                builder.int8_mode = True
                builder.fp16_mode = True
                builder.strict_type_constraints = strict_datatype

        valid_config(config)

        # Parse onnx model
        with open(model_file, 'rb') as model:
            if not parser.parse(model.read()):
                logger.error('ERROR: Fail to parse the ONNX file.')
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                return None

        if calib is not None:
            if trt_version == TRT8:
                trt_config.int8_calibrator = calib
            else:
                builder.int8_calibrator = calib
            # This design may not be correct if output more than one
            for i in range(network.num_layers):
                if config is None:
                    break
                layer = network.get_layer(i)
                if layer.name in config:
                    w_bits = config[layer.name]['weight_bits']
                    a_bits = config[layer.name]['output_bits']
                    layer.precision = Precision_Dict[w_bits]
                    layer.set_output_type(0, Precision_Dict[a_bits])
        else:
            # This implementation may be incorrect when output number > 1
            for i in range(network.num_layers):
                if config is None:
                    # no low bits layer need to be set, keep original model
                    break
                layer = network.get_layer(i)
                if layer.name not in config:
                    continue
                # layer numbers of gemm changed during pytorch->onnx model convertion, need special handle
                if layer.name[0:4] == "Gemm":
                    handle_gemm(network, i, config)
                    continue

                # If weight_bits exists in config, set layer precision and layer's input tensor dynamic range.
                if 'weight_bits' in config[layer.name]:
                    assert 'tracked_min_input' in config[layer.name]
                    assert 'tracked_max_input' in config[layer.name]
                    w_bits = config[layer.name]['weight_bits']
                    tracked_min_input = config[layer.name]['tracked_min_input']
                    tracked_max_input = config[layer.name]['tracked_max_input']
                    layer.precision = Precision_Dict[w_bits]
                    in_tensor = layer.get_input(0)
                    in_tensor.dynamic_range = (tracked_min_input, tracked_max_input)

                # If output exists in config, set layer output type and layer's output tensor dynamic range.
                if 'output_bits' in config[layer.name]:
                    assert 'tracked_min_output' in config[layer.name]
                    assert 'tracked_max_output' in config[layer.name]
                    a_bits = config[layer.name]['output_bits']
                    tracked_min_output = config[layer.name]['tracked_min_output']
                    tracked_max_output = config[layer.name]['tracked_max_output']
                    layer.set_output_type(0, Precision_Dict[a_bits])
                    out_tensor = layer.get_output(0)
                    out_tensor.dynamic_range = (tracked_min_output, tracked_max_output)

        # Build engine and do int8 calibration.
        if trt_version == TRT8:
            engine = builder.build_engine(network, trt_config)
        else:
            engine = builder.build_cuda_engine(network)
        return engine

class ModelSpeedupTensorRT(BaseModelSpeedup):
    def __init__(self, model, input_shape, config=None, onnx_path="default_model.onnx", extra_layer_bits=32, strict_datatype=True,
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
            Config recording bits number and name of layers.
        onnx_path : str
            The path user want to store onnx model which is converted from pytorch model.
        extra_layer_bits : int
            Other layers which are not in config will be quantized to corresponding bits number.
        strict_datatype : bool
            Whether constrain layer bits to the number given in config or not. If true, all the layer
            will be set to given bits strictly. Otherwise, these layers will be set automatically by
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
        self.extra_layer_bits = extra_layer_bits
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
        """
        Convert pytorch tensor to numpy darray

        Parameters
        ----------
        onnx_path : str
            The path of onnx model

        Returns
        -------
        tensorrt.IExecutionContext
            Context for executing inference using an ICudaEngine
        """
        calib_data = None
        if type(self.calib_data_loader) == torch.utils.data.dataloader.DataLoader:
            calib_data_set = []
            for data, _ in self.calib_data_loader:
                calib_data_set.append(data)
            calib_data = np.concatenate(calib_data_set)
        elif type(self.calib_data_loader) == torch.Tensor:
            # trt need numpy as calibration data, only cpu data can convert to numpy directly
            if self.calib_data_loader.device != torch.device("cpu"):
                self.calib_data_loader = self.calib_data_loader.to("cpu")
            calib_data = self.calib_data_loader.numpy()
        else:
            raise ValueError("Not support calibration datatype")
        calib = calibrator.Calibrator(calib_data, self.calibration_cache, self.batchsize, self.calibrate_type)

        # build inference engine with calibration
        engine = build_engine(onnx_path, self.onnx_config, self.extra_layer_bits, self.strict_datatype, calib)
        return engine.create_execution_context()

    def _tensorrt_build_withoutcalib(self, onnx_path):
        """
        Build inference engine without calibration

        Parameters
        ----------
        onnx_path : str
            The path of onnx model

        Returns
        -------
        tensorrt.IExecutionContext
            Context for executing inference using an ICudaEngine
        """
        engine = build_engine(onnx_path, self.onnx_config, self.extra_layer_bits, self.strict_datatype)
        return engine.create_execution_context()

    def inference(self, test_data):
        """
        Do inference by tensorrt builded engine.

        Parameters
        ----------
        test_data : pytorch tensor
            Model input tensor
        """
        # convert pytorch tensor to numpy darray
        if test_data.device != torch.device("cpu"):
            test_data = test_data.to("cpu")
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
        """
        Export TensorRT quantized model engine which only can be loaded by TensorRT deserialize API.

        Parameters
        ----------
        path : str
            The path of export model
        """
        assert path is not None
        with open(path, "wb") as f:
            f.write(self.context.engine.serialize())
            logger.info("TensorRT engine has been saved to %s", path)

    def load_quantized_model(self, path):
        """
        Load TensorRT quantized model engine from specific path.

        Parameters
        ----------
        path : str
            The path of export model
        """
        assert path is not None
        with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            self.context = engine.create_execution_context()
            logger.info("Load TensorRT engine from %s successfully.", path)