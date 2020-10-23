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

def update_ema(biased_ema, value, decay, step):
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
    step : int
        current step

    Returns
    -------
    float, float
    """
    biased_ema = biased_ema * decay + (1 - decay) * value
    unbiased_ema = biased_ema / (1 - decay ** step)  # Bias correction
    return biased_ema, unbiased_ema


def update_quantization_param(bits, rmin, rmax):
    """
    calculate the `zero_point` and `scale`.

    Parameters
    ----------
    bits : int
        quantization bits length
    rmin : float
        min value of real value
    rmax : float
        max value of real value

    Returns
    -------
    float, float
    """
    # extend the [min, max] interval to ensure that it contains 0.
    # Otherwise, we would not meet the requirement that 0 be an exactly
    # representable value.
    rmin = min(rmin, 0)
    rmax = max(rmax, 0)

    # the min and max quantized values, as floating-point values
    qmin = 0
    qmax = (1 << bits) - 1
    # First determine the scale.
    scale = (rmax - rmin) / (qmax - qmin)

    # Zero-point computation.
    initial_zero_point = qmin - rmin / scale

    # Now we need to nudge the zero point to be an integer
    nudged_zero_point = 0
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
        self.steps = 1
        modules_to_compress = self.get_modules_to_compress()
        for layer, config in modules_to_compress:
            layer.module.register_buffer("zero_point", None)
            layer.module.register_buffer("scale", None)
            if "output" in config.get("quant_types", []):
                layer.module.register_buffer('ema_decay', torch.Tensor([0.99]))
                layer.module.register_buffer('tracked_min_biased', torch.zeros(1))
                layer.module.register_buffer('tracked_min', torch.zeros(1))
                layer.module.register_buffer('tracked_max_biased', torch.zeros(1))
                layer.module.register_buffer('tracked_max', torch.zeros(1))

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
        real_val : float
            real value to be quantized

        Returns
        -------
        float
        """
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
        weight = copy.deepcopy(wrapper.module.old_weight.data)
        weight_bits = get_bits_length(config, 'weight')
        quant_start_step = config.get('quant_start_step', 0)
        assert weight_bits >= 1, "quant bits length should be at least 1"

        if quant_start_step > self.steps:
            return weight

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
        wrapper.module.weight = weight
        return weight

    def quantize_output(self, output, wrapper, **kwargs):
        config = wrapper.config
        module = wrapper.module
        output_bits = get_bits_length(config, 'output')
        quant_start_step = config.get('quant_start_step', 0)
        assert output_bits >= 1, "quant bits length should be at least 1"

        if quant_start_step > self.steps:
            return output

        current_min, current_max = torch.min(output), torch.max(output)
        module.tracked_min_biased, module.tracked_min = update_ema(module.tracked_min_biased, current_min,
                                                                   module.ema_decay, self.steps)
        module.tracked_max_biased, module.tracked_max = update_ema(module.tracked_max_biased, current_max,
                                                                   module.ema_decay, self.steps)
        module.scale, module.zero_point = update_quantization_param(output_bits, module.tracked_min, module.tracked_max)
        out = self._quantize(output_bits, module, output)
        out = self._dequantize(module, out)
        return out

    def fold_bn(self, config, **kwargs):
        # TODO simulate folded weight
        pass

    def step_with_optimizer(self):
        """
        override `compressor` `step` method, quantization only happens after certain number of steps
        """
        self.steps += 1


class DoReFaQuantizer(Quantizer):
    """Quantizer using the DoReFa scheme, as defined in:
    Zhou et al., DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients
    (https://arxiv.org/abs/1606.06160)
    """

    def __init__(self, model, config_list, optimizer=None):
        super().__init__(model, config_list, optimizer)

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
        # wrapper.module.weight.data = weight
        return weight

    def quantize(self, input_ri, q_bits):
        scale = pow(2, q_bits) - 1
        output = torch.round(input_ri * scale) / scale
        return output


class ClipGrad(QuantGrad):
    @staticmethod
    def quant_backward(tensor, grad_output, quant_type):
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
        return weight

    def quantize_output(self, output, wrapper, **kwargs):
        out = torch.sign(output)
        # remove zeros
        out[out == 0] = 1
        return out
