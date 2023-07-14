# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
import logging
import tensorrt as trt
import numpy as np
import torch

from . import frontend_to_onnx as fonnx
from . import trt_pycuda as common # NOTE pycuda becomes a dependency, consider adding it to dependencies
from .backend import BaseModelSpeedup

TRT8 = 8
TRT_LOGGER = trt.Logger()
logger = logging.getLogger(__name__)

Precision_Dict = {
    8: trt.int8,
    16: trt.float16,
    # NOTE: uncomment them or refactor when they are required
    # 'f32': trt.float32,
    # 'i32': trt.int32
    # trt.bool
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

def print_layer_precisions(network):
    print('The layer precisions and dynamic ranges are:')
    for layer_idx in range(network.num_layers):
        layer = network.get_layer(layer_idx)
        out = layer.get_output(0)
        print(layer.name, layer.precision, out.dynamic_range)

def _handle_gemm(layer, config, out2layer, in2layer):
    """
    Gemm is special case. the following is the graph structure of Gemm in trt's graph
    input                       ->| Gemm  ->| ElementWise
    LayerType.Constant (weight) ->|
    LayerType.Constant (bias) -> Shuffle  ->|
    assume quantize input, output, and weight
    """
    w_bits = config['weight_bits']
    layer.precision = Precision_Dict[w_bits]
    # handle the input tensor
    in_tensor = layer.get_input(0)
    in_tensor.dynamic_range = (config['tracked_min_input'], config['tracked_max_input'])
    # handle the output tensor
    out_tensor = layer.get_output(0)
    out_tensor.dynamic_range = (config['tracked_min_output'], config['tracked_max_output'])
    # handle weight
    w_in_tensor = layer.get_input(1)
    weight_layer = out2layer[w_in_tensor.name]
    assert weight_layer.type == trt.LayerType.CONSTANT
    weight_layer.precision = Precision_Dict[w_bits]
    weight_layer.set_output_type(0, Precision_Dict[w_bits])
    w_out_tensor = weight_layer.get_output(0)
    w_out_tensor.dynamic_range = (config['min_weight'], config['max_weight'])
    print('special gemm: ', w_out_tensor.dynamic_range)
    # TODO: handle sum & bias
    # NOTE: a feasible way is setting bias to 0 in quantization algorithm size
    # and track the dynamic range without bias.
    return weight_layer.name

def apply_precision_to_layer(layer, config):
    if 'weight_bits' in config:
        w_bits = config['weight_bits']
        layer.precision = Precision_Dict[w_bits]
    if 'input_bits' in config:
        assert 'tracked_min_input' in config
        assert 'tracked_max_input' in config
        tracked_min_input = config['tracked_min_input']
        tracked_max_input = config['tracked_max_input']
        # NOTE: only support one input tensor for now
        in_tensor = layer.get_input(0)
        in_tensor.dynamic_range = (tracked_min_input, tracked_max_input)
    if 'output_bits' in config:
        assert 'tracked_min_output' in config
        assert 'tracked_max_output' in config
        act_bits = config['output_bits']
        tracked_min_output = config['tracked_min_output']
        tracked_max_output = config['tracked_max_output']
        layer.set_output_type(0, Precision_Dict[act_bits])
        out_tensor = layer.get_output(0)
        out_tensor.dynamic_range = (tracked_min_output, tracked_max_output)

def propagate_from_low_bit_predecessor(layer, out2layer, default_precision=trt.float16):
    """
    Returns
    -------
    layer precision
        current layer's precision
    (min, max)
        dynamic range of current layer's output tensor
    """
    dynamic_range = None
    tensor = layer.get_input(0)
    if tensor is not None:
        predecessor = out2layer[tensor.name]
        # NOTE: only support int8 for now
        if predecessor.get_output_type(0) == trt.int8:
            dynamic_range = tensor.dynamic_range

    if layer.name[0:4] == 'Relu':
        assert dynamic_range is not None
        return trt.int8, (0, dynamic_range[1])
    elif layer.name[0:3] == 'Add':
        #assert dynamic_range is not None
        return trt.int32, None
    else:
        logger.warning(f'set op {layer.name} to default precision {default_precision}')
        return default_precision, None

def config_network_precision(network, config):
    """
    The idea here is that ...
    TODO: make sure the weights are the ones after quantize and dequantize.
    In the network, bn has been folded by trt OnnxParser
    """
    # build two auxiliary indices
    out2layer = {}
    in2layer = {}
    for layer_idx in range(network.num_layers):
        layer = network.get_layer(layer_idx)
        for i in range(layer.num_outputs):
            output = layer.get_output(i)
            out2layer[output.name] = layer
        for i in range(layer.num_inputs):
            _input = layer.get_input(i)
            if _input.name in in2layer:
                in2layer[_input.name].append(layer)
            else:
                in2layer[_input.name] = [layer]

    net_input = network.get_input(0)
    assert net_input.name in in2layer

    # traverse the network/graph and specify precision and dynamic range
    for layer_idx in range(network.num_layers):
        # assume the traverse order is topological
        layer = network.get_layer(layer_idx)
        if layer.name in config:
            if layer.name[0:4] == 'Gemm':
                _handle_gemm(layer, config[layer.name], out2layer, in2layer)
            else:
                apply_precision_to_layer(layer, config[layer.name])
        else:
            precision, dynamic_range = propagate_from_low_bit_predecessor(layer, out2layer)
            if precision:
                layer.precision = precision
                layer.set_output_type(0, precision)
            if dynamic_range:
                out_tensor = layer.get_output(0)
                out_tensor.dynamic_range = dynamic_range

    print_layer_precisions(network)

def build_engine_without_calib(onnx_model_file, config):
    """
    This function builds an engine from an onnx model following the precisions
    and dynamic range in config without calibrator.

    Parameters
    ----------
    onnx_model_file : str
        The path of onnx model
    config : dict
        Config recording bits number and name of layers

    Returns
    -------
    tensorrt.ICudaEngine
        An ICudaEngine for executing inference on a built network
    """
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(common.explicit_batch())
    trt_config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    builder.max_batch_size = 1 # TODO: check whether it is necessary

    trt_config.max_workspace_size = common.GiB(4)

    trt_config.set_flag(trt.BuilderFlag.INT8)
    trt_config.set_flag(trt.BuilderFlag.FP16)
    trt_config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

    # Parse onnx model
    with open(onnx_model_file, 'rb') as model:
        if not parser.parse(model.read()):
            logger.error('ERROR: Fail to parse the ONNX file.')
            for error in range(parser.num_errors):
                logger.error(parser.get_error(error))
            raise ValueError('Failed to parse the ONNX file.')

    config_network_precision(network, config)

    # Build engine and do int8 calibration.
    engine = builder.build_engine(network, trt_config)
    return engine

def config_network_to_int8(network):
    for layer_idx in range(network.num_layers):
        layer = network.get_layer(layer_idx)
        layer.precision = trt.int8

def build_engine_with_calib(onnx_model_file, calib, input_shape):
    """
    Parameters
    ----------
    """
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(common.explicit_batch())
    trt_config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    builder.max_batch_size = input_shape[0]
    trt_config.max_workspace_size = common.GiB(8)
    trt_config.set_flag(trt.BuilderFlag.INT8)
    trt_config.set_flag(trt.BuilderFlag.FP16)
    trt_config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    trt_config.int8_calibrator = calib

    with open(onnx_model_file, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                TRT_LOGGER.log(TRT_LOGGER.ERROR, parser.get_error(error))
            raise ValueError('Failed to parse the ONNX file.')

    TRT_LOGGER.log(TRT_LOGGER.INFO, f'input number: {network.num_inputs}')
    TRT_LOGGER.log(TRT_LOGGER.INFO, f'output number: {network.num_outputs}')

    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(input_name, min=input_shape, opt=input_shape, max=input_shape)
    trt_config.add_optimization_profile(profile)

    config_network_to_int8(network) # not sure whether it is necessary because trt.BuilderFlag.INT8 is set.

    engine = builder.build_engine(network, trt_config)
    return engine

class ModelSpeedupTensorRT(BaseModelSpeedup):
    """
    Parameters
    ----------
    model : pytorch model
        The model to speedup by quantization.
    input_shape : tuple
        The input shape of the model, shall pass it to torch.onnx.export.
        Note, the batch size of input_shape is the inference batch of the created trt engine,
        it should be equal to the batch size of running test with the engine.
    config : dict
        Config recording bits number and name of layers.
    onnx_path : str
        The path user want to store onnx model which is converted from pytorch model.
    """

    def __init__(self, model, input_shape, config=None, onnx_path="default_model.onnx"):
        super().__init__(model, config)
        self.model = model
        self.input_shape = input_shape
        self.config = config
        self.onnx_path = onnx_path
        # Input name of onnx model providing for torch.onnx.export to generate onnx model
        # Output name of onnx model providing for torch.onnx.export to generate onnx model
        self.input_names = ["actual_input_1"]
        self.output_names = ["output1"]

        self.engine = None
        self.context = None
        self.inputs = None
        self.outputs = None
        self.bindings = None
        self.stream = None

        trt_version = int(trt.__version__[0])
        assert trt_version >= TRT8, "Version of TensorRT is too old, please \
            update TensorRT to version >= 8.0"

    def compress(self):
        """
        This speedup approach uses ```trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS``` mode of trt engine,
        which means it would faithfully enforce the precisions and dynamic ranges
        in user passed-in config i.e., self.config.
        Thus, users must provide dynamic range for all tensors that are not Int32 or Bool.
        """
        assert self.config is not None
        # Convert pytorch model to onnx model and save onnx model in onnx_path
        _, onnx_config = fonnx.torch_to_onnx(self.model, self.config, input_shape=self.input_shape,
            model_path=self.onnx_path, input_names=self.input_names, output_names=self.output_names)
        valid_config(onnx_config)
        self.engine = build_engine_without_calib(self.onnx_path, onnx_config)

    def compress_with_calibrator(self, calib):
        """
        This speedup approach leverages calibrator
        """
        # convert model to onnx
        device = torch.device('cpu')
        dummy_input = torch.randn(self.input_shape).to(device)
        self.model.to(device)
        torch.onnx.export(self.model, dummy_input, self.onnx_path, verbose=False,
            input_names=self.input_names, output_names=self.output_names, export_params=True)
        # build endine
        self.engine = build_engine_with_calib(self.onnx_path, calib, self.input_shape)

    def inference(self, test_data, reset_context=False):
        """
        Do inference by tensorrt builded engine.
        Note, the batch size of test_data should be equal to the batch size used in building the engine.

        Parameters
        ----------
        test_data : pytorch tensor
            Model input tensor, the first dimension should be batch dimension.
        reset_context : bool
            whether reset the engine context.

        Returns
        -------
        torch.Tensor
            the flattened tensor (Note, this value may be changed after the next inference).
        float
            the time span of the inference
        """
        if self.context is None or reset_context:
            self.context = self.engine.create_execution_context()
            self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)
            self.context.set_optimization_profile_async(0, self.stream.handle)

        engine_input_shape = self.engine.get_binding_shape(0)
        assert engine_input_shape[0] == test_data.size()[0]
        if test_data.device != torch.device('cpu'):
            logger.warning('test_data should be placed on CPU.')
            test_data = test_data.to(torch.device('cpu'))
        test_data = test_data.numpy()
        assert test_data.dtype == np.float32

        np.copyto(self.inputs[0].host, test_data.ravel())
        start_time = time.time()
        trt_outputs = common.do_inference_v2(self.context, bindings=self.bindings, inputs=self.inputs,
                                                outputs=self.outputs, stream=self.stream)
        time_span = time.time() - start_time
        return torch.as_tensor(trt_outputs[0]), time_span

    def export_quantized_model(self, path):
        """
        Export TensorRT quantized model engine which only can be loaded by TensorRT deserialize API.

        Parameters
        ----------
        path : str
            The path of export model
        """
        pass

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