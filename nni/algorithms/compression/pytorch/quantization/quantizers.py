# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import copy
import torch
from schema import Schema, And, Or, Optional
from nni.compression.pytorch.utils.config_validation import CompressorSchema
from nni.compression.pytorch.compressor import Quantizer, QuantGrad, QuantType

__all__ = ['NaiveQuantizer', 'QAT_Quantizer', 'DoReFaQuantizer', 'BNNQuantizer']

logger = logging.getLogger(__name__)


class NaiveQuantizer(Quantizer):
    """quantize weight to 8 bits
    """

    def __init__(self, model, config_list, optimizer=None):
        super().__init__(model, config_list, optimizer)
        self.layer_scale = {}

    def validate_config(self, model, config_list):
        schema = CompressorSchema([{
            Optional('quant_types'): ['weight'],
            Optional('quant_bits'): Or(8, {'weight': 8}),
            Optional('op_types'): [str],
            Optional('op_names'): [str]
        }], model, logger)

        schema.validate(config_list)

    def quantize_weight(self, wrapper, **kwargs):
        weight = copy.deepcopy(wrapper.module.old_weight.data)
        new_scale = weight.abs().max() / 127
        scale = max(self.layer_scale.get(wrapper.name, 0), new_scale)
        self.layer_scale[wrapper.name] = scale
        orig_type = weight.type()  # TODO: user layer
        weight = weight.div(scale).type(torch.int8).type(orig_type).mul(scale)
        wrapper.module.weight = weight
        return weight

def update_ema(biased_ema, value, decay):
    """
    calculate biased stat and unbiased stat in each step using exponential moving average method

    Parameters
    ----------
    biased_ema : float
        previous stat value
    value : float
        current stat value
    decay : float
        the weight of previous stat value, larger means smoother curve

    Returns
    -------
    float, float
    """
    biased_ema = biased_ema * decay + (1 - decay) * value
    return biased_ema 


def update_quantization_param(bits, rmin, rmax):
    """
    calculate the `zero_point` and `scale`.

    Parameters
    ----------
    bits : int
        quantization bits length
    rmin : Tensor
        min value of real value
    rmax : Tensor
        max value of real value

    Returns
    -------
    float, float
    """
    # extend the [min, max] interval to ensure that it contains 0.
    # Otherwise, we would not meet the requirement that 0 be an exactly
    # representable value.
    rmin = torch.min(rmin, torch.Tensor([0]).to(rmin.device))
    rmax = torch.max(rmax, torch.Tensor([0]).to(rmin.device))
    qmin = torch.Tensor([0]).to(rmin.device)
    qmax = torch.Tensor([(1 << bits) - 1]).to(rmin.device)

    # First determine the scale.
    scale = (rmax - rmin) / (qmax - qmin)

    # Zero-point computation.
    initial_zero_point = qmin - rmin / scale

    # Now we need to nudge the zero point to be an integer
    if initial_zero_point < qmin:
        nudged_zero_point = qmin
    elif initial_zero_point > qmax:
        nudged_zero_point = qmax
    else:
        nudged_zero_point = torch.round(initial_zero_point)

    return scale, nudged_zero_point


def get_bits_length(config, quant_type):
    if isinstance(config["quant_bits"], int):
        return config["quant_bits"]
    else:
        return config["quant_bits"].get(quant_type)

class QATGrad(QuantGrad):
    @staticmethod
    def quant_backward(tensor, grad_output, quant_type, scale, zero_point, qmin, qmax):
        tensor_q = QuantGrad._quantize(tensor, scale, zero_point)
        mask = (tensor_q < qmin) | (tensor_q > qmax)
        grad_output[mask] = 0
        return grad_output


class QAT_Quantizer(Quantizer):
    """Quantizer defined in:
    Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference
    http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf
    """

    def __init__(self, model, config_list, optimizer=None):
        """
        Parameters
        ----------
        layer : LayerInfo
            the layer to quantize
        config_list : list of dict
            list of configurations for quantization
            supported keys for dict:
                - quant_types : list of string
                    type of quantization you want to apply, currently support 'weight', 'input', 'output'
                - quant_bits : int or dict of {str : int}
                    bits length of quantization, key is the quantization type, value is the length, eg. {'weight', 8},
                    when the type is int, all quantization types share same bits length
                - quant_start_step : int
                    disable quantization until model are run by certain number of steps, this allows the network to enter a more stable
                    state where activation quantization ranges do not exclude a signiﬁcant fraction of values, default value is 0
                - op_types : list of string
                    types of nn.module you want to apply quantization, eg. 'Conv2d'
        """
        super().__init__(model, config_list, optimizer)
        self.quant_grad = QATGrad
        modules_to_compress = self.get_modules_to_compress()
        self.bound_model.register_buffer("steps", torch.Tensor([1]))
        for layer, config in modules_to_compress:
            layer.module.register_buffer("zero_point", torch.Tensor([0.0]))
            layer.module.register_buffer("scale", torch.Tensor([1.0]))
            layer.module.register_buffer('ema_decay', torch.Tensor([0.99]))
            if "weight" in config.get("quant_types", []):
                layer.module.register_buffer('weight_bit', torch.zeros(1))
                layer.module.register_buffer('tracked_min_input', torch.zeros(1))
                layer.module.register_buffer('tracked_max_input', torch.zeros(1))
            if "output" in config.get("quant_types", []):
                layer.module.register_buffer('activation_bit', torch.zeros(1))
                layer.module.register_buffer('tracked_min_activation', torch.zeros(1))
                layer.module.register_buffer('tracked_max_activation', torch.zeros(1))
                

    def _del_simulated_attr(self, module):
        """
        delete redundant parameters in quantize module
        """
        del_attr_list = ['old_weight', 'ema_decay', 'tracked_min_activation', 'tracked_max_activation', 'tracked_min_input', \
        'tracked_max_input', 'scale', 'zero_point', 'weight_bit', 'activation_bit']
        for attr in del_attr_list:
            if hasattr(module, attr):
                delattr(module, attr)

    def validate_config(self, model, config_list):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Model to be pruned
        config_list : list of dict
            List of configurations
        """
        schema = CompressorSchema([{
            Optional('quant_types'): Schema([lambda x: x in ['weight', 'output']]),
            Optional('quant_bits'): Or(And(int, lambda n: 0 < n < 32), Schema({
                Optional('weight'): And(int, lambda n: 0 < n < 32),
                Optional('output'): And(int, lambda n: 0 < n < 32),
            })),
            Optional('quant_start_step'): And(int, lambda n: n >= 0),
            Optional('op_types'): [str],
            Optional('op_names'): [str]
        }], model, logger)

        schema.validate(config_list)

    def _quantize(self, bits, op, real_val):
        """
        quantize real value.

        Parameters
        ----------
        bits : int
            quantization bits length
        op : torch.nn.Module
            target module
        real_val : Tensor
            real value to be quantized

        Returns
        -------
        Tensor
        """
        op.zero_point = op.zero_point.to(real_val.device)
        op.scale = op.scale.to(real_val.device)
        transformed_val = op.zero_point + real_val / op.scale
        qmin = 0
        qmax = (1 << bits) - 1
        clamped_val = torch.clamp(transformed_val, qmin, qmax)
        quantized_val = torch.round(clamped_val)
        return quantized_val

    def _dequantize(self, op, quantized_val):
        """
        dequantize quantized value.
        Because we simulate quantization in training process, all the computations still happen as float point computations, which means we
        first quantize tensors then dequantize them. For more details, please refer to the paper.

        Parameters
        ----------
        op : torch.nn.Module
            target module
        quantized_val : float
            quantized_val value to be dequantized

        Returns
        -------
        float
        """
        real_val = op.scale * (quantized_val - op.zero_point)
        return real_val

    def quantize_weight(self, wrapper, **kwargs):
        config = wrapper.config
        module = wrapper.module
        input = kwargs['input_tensor']
        weight = copy.deepcopy(wrapper.module.old_weight.data)
        weight_bits = get_bits_length(config, 'weight')
        quant_start_step = config.get('quant_start_step', 0)
        assert weight_bits >= 1, "quant bits length should be at least 1"

        # we dont update weight in evaluation stage
        if quant_start_step > self.bound_model.steps or not wrapper.training:
            module.tracked_min_input, module.tracked_max_input = torch.min(input), torch.max(input)
            return weight

        if wrapper.training:
            current_min, current_max = torch.min(input), torch.max(input)
            module.tracked_min_input = update_ema(module.tracked_min_input, current_min,
                                                                       module.ema_decay)
            module.tracked_max_input = update_ema(module.tracked_max_input, current_max,
                                                                       module.ema_decay)

        # if bias exists, quantize bias to uint32
        if hasattr(wrapper.module, 'bias') and wrapper.module.bias is not None:
            bias = wrapper.module.bias.data
            bias_bits = 32
            rmin, rmax = torch.min(bias), torch.max(bias)
            module.scale, module.zero_point = update_quantization_param(bias_bits, rmin, rmax)
            bias = self._quantize(bias_bits, module, bias)
            bias = self._dequantize(module, bias)
            wrapper.module.bias.data = bias


        # quantize weight
        rmin, rmax = torch.min(weight), torch.max(weight)
        module.scale, module.zero_point = update_quantization_param(weight_bits, rmin, rmax)
        weight = self._quantize(weight_bits, module, weight)
        weight = self._dequantize(module, weight)
        module.weight_bit = torch.Tensor([weight_bits])
        wrapper.module.weight = weight
        return weight

    def quantize_output(self, output, wrapper, **kwargs):
        config = wrapper.config
        module = wrapper.module
        output_bits = get_bits_length(config, 'output')
        module.activation_bit = torch.Tensor([output_bits])
        quant_start_step = config.get('quant_start_step', 0)
        assert output_bits >= 1, "quant bits length should be at least 1"

        if quant_start_step > self.bound_model.steps:
            module.tracked_min_activation, module.tracked_max_activation = torch.min(output), torch.max(output)
            return output

        # we dont update output quantization parameters in evaluation stage
        if wrapper.training:
            current_min, current_max = torch.min(output), torch.max(output)
            module.tracked_min_activation = update_ema(module.tracked_min_activation, current_min,
                                                                       module.ema_decay)
            module.tracked_max_activation = update_ema(module.tracked_max_activation, current_max,
                                                                       module.ema_decay)
            module.scale, module.zero_point = update_quantization_param(output_bits, module.tracked_min_activation, module.tracked_max_activation)
        out = self._quantize(output_bits, module, output)
        out = self._dequantize(module, out)
        return out

    def export_model(self, model_path, calibration_path=None, onnx_path=None, input_shape=None, device=None):
        """
        Export quantized model weights and calibration parameters(optional)

        Parameters
        ----------
        model_path : str
            path to save quantized model weight
        calibration_path : str
            (optional) path to save quantize parameters after calibration
        onnx_path : str
            (optional) path to save onnx model
        input_shape : list or tuple
            input shape to onnx model
        device : torch.device
            device of the model, used to place the dummy input tensor for exporting onnx file.
            the tensor is placed on cpu if ```device``` is None

        Returns
        -------
        Dict
        """
        assert model_path is not None, 'model_path must be specified'
        self._unwrap_model()
        calibration_config = {}

        for name, module in self.bound_model.named_modules():
            if hasattr(module, 'weight_bit') or hasattr(module, 'activation_bit'):
                calibration_config[name] = {}
            if hasattr(module, 'weight_bit'):
                calibration_config[name]['weight_bit'] = int(module.weight_bit)
                calibration_config[name]['tracked_min_input'] = float(module.tracked_min_input)
                calibration_config[name]['tracked_max_input'] = float(module.tracked_max_input)
            if hasattr(module, 'activation_bit'):
                calibration_config[name]['activation_bit'] = int(module.activation_bit)
                calibration_config[name]['tracked_min_activation'] = float(module.tracked_min_activation)
                calibration_config[name]['tracked_max_activation'] = float(module.tracked_max_activation)
            self._del_simulated_attr(module)

        self.export_model_save(self.bound_model, model_path, calibration_config, calibration_path, onnx_path, input_shape, device)

        return calibration_config

    def fold_bn(self, config, **kwargs):
        # TODO simulate folded weight
        pass

    def step_with_optimizer(self):
        """
        override `compressor` `step` method, quantization only happens after certain number of steps
        """
        self.bound_model.steps +=1


class DoReFaQuantizer(Quantizer):
    """Quantizer using the DoReFa scheme, as defined in:
    Zhou et al., DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients
    (https://arxiv.org/abs/1606.06160)
    """

    def __init__(self, model, config_list, optimizer=None):
        super().__init__(model, config_list, optimizer)
        modules_to_compress = self.get_modules_to_compress()
        for layer, config in modules_to_compress:
            if "weight" in config.get("quant_types", []):
                layer.module.register_buffer('weight_bit', torch.zeros(1))

    def _del_simulated_attr(self, module):
        """
        delete redundant parameters in quantize module
        """
        del_attr_list = ['old_weight', 'weight_bit']
        for attr in del_attr_list:
            if hasattr(module, attr):
                delattr(module, attr)

    def validate_config(self, model, config_list):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Model to be pruned
        config_list : list of dict
            List of configurations
        """
        schema = CompressorSchema([{
            Optional('quant_types'): Schema([lambda x: x in ['weight']]),
            Optional('quant_bits'): Or(And(int, lambda n: 0 < n < 32), Schema({
                Optional('weight'): And(int, lambda n: 0 < n < 32)
            })),
            Optional('op_types'): [str],
            Optional('op_names'): [str]
        }], model, logger)

        schema.validate(config_list)

    def quantize_weight(self, wrapper, **kwargs):
        weight = copy.deepcopy(wrapper.module.old_weight.data)
        weight_bits = get_bits_length(wrapper.config, 'weight')
        weight = weight.tanh()
        weight = weight / (2 * weight.abs().max()) + 0.5
        weight = self.quantize(weight, weight_bits)
        weight = 2 * weight - 1
        wrapper.module.weight = weight
        wrapper.module.weight_bit = torch.Tensor([weight_bits])
        # wrapper.module.weight.data = weight
        return weight

    def quantize(self, input_ri, q_bits):
        scale = pow(2, q_bits) - 1
        output = torch.round(input_ri * scale) / scale
        return output

    def export_model(self, model_path, calibration_path=None, onnx_path=None, input_shape=None, device=None):
        """
        Export quantized model weights and calibration parameters(optional)

        Parameters
        ----------
        model_path : str
            path to save quantized model weight
        calibration_path : str
            (optional) path to save quantize parameters after calibration
        onnx_path : str
            (optional) path to save onnx model
        input_shape : list or tuple
            input shape to onnx model
        device : torch.device
            device of the model, used to place the dummy input tensor for exporting onnx file.
            the tensor is placed on cpu if ```device``` is None

        Returns
        -------
        Dict
        """
        assert model_path is not None, 'model_path must be specified'
        self._unwrap_model()
        calibration_config = {}

        for name, module in self.bound_model.named_modules():
            if hasattr(module, 'weight_bit'):
                calibration_config[name] = {}
                calibration_config[name]['weight_bit'] = int(module.weight_bit)
            self._del_simulated_attr(module)

        self.export_model_save(self.bound_model, model_path, calibration_config, calibration_path, onnx_path, input_shape, device)

        return calibration_config


class ClipGrad(QuantGrad):
    @staticmethod
    def quant_backward(tensor, grad_output, quant_type, scale, zero_point, qmin, qmax):
        if quant_type == QuantType.QUANT_OUTPUT:
            grad_output[torch.abs(tensor) > 1] = 0
        return grad_output


class BNNQuantizer(Quantizer):
    """Binarized Neural Networks, as defined in:
    Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1
    (https://arxiv.org/abs/1602.02830)
    """

    def __init__(self, model, config_list, optimizer=None):
        super().__init__(model, config_list, optimizer)
        self.quant_grad = ClipGrad
        modules_to_compress = self.get_modules_to_compress()
        for layer, config in modules_to_compress:
            if "weight" in config.get("quant_types", []):
                layer.module.register_buffer('weight_bit', torch.zeros(1))

    def _del_simulated_attr(self, module):
        """
        delete redundant parameters in quantize module
        """
        del_attr_list = ['old_weight', 'weight_bit']
        for attr in del_attr_list:
            if hasattr(module, attr):
                delattr(module, attr)

    def validate_config(self, model, config_list):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Model to be pruned
        config_list : list of dict
            List of configurations
        """
        schema = CompressorSchema([{
            Optional('quant_types'): Schema([lambda x: x in ['weight', 'output']]),
            Optional('quant_bits'): Or(And(int, lambda n: 0 < n < 32), Schema({
                Optional('weight'): And(int, lambda n: 0 < n < 32),
                Optional('output'): And(int, lambda n: 0 < n < 32),
            })),
            Optional('op_types'): [str],
            Optional('op_names'): [str]
        }], model, logger)

        schema.validate(config_list)

    def quantize_weight(self, wrapper, **kwargs):
        weight = copy.deepcopy(wrapper.module.old_weight.data)
        weight = torch.sign(weight)
        # remove zeros
        weight[weight == 0] = 1
        wrapper.module.weight = weight
        wrapper.module.weight_bit = torch.Tensor([1.0])
        return weight

    def quantize_output(self, output, wrapper, **kwargs):
        out = torch.sign(output)
        # remove zeros
        out[out == 0] = 1
        return out

    def export_model(self, model_path, calibration_path=None, onnx_path=None, input_shape=None, device=None):
        """
        Export quantized model weights and calibration parameters(optional)

        Parameters
        ----------
        model_path : str
            path to save quantized model weight
        calibration_path : str
            (optional) path to save quantize parameters after calibration
        onnx_path : str
            (optional) path to save onnx model
        input_shape : list or tuple
            input shape to onnx model
        device : torch.device
            device of the model, used to place the dummy input tensor for exporting onnx file.
            the tensor is placed on cpu if ```device``` is None

        Returns
        -------
        Dict
        """
        assert model_path is not None, 'model_path must be specified'
        self._unwrap_model()
        calibration_config = {}

        for name, module in self.bound_model.named_modules():
            if hasattr(module, 'weight_bit'):
                calibration_config[name] = {}
                calibration_config[name]['weight_bit'] = int(module.weight_bit)
            self._del_simulated_attr(module)

        self.export_model_save(self.bound_model, model_path, calibration_config, calibration_path, onnx_path, input_shape, device)

        return calibration_config