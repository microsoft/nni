import logging
import torch
from .compressor import Quantizer

__all__ = ['NaiveQuantizer', 'QAT_Quantizer', 'DoReFaQuantizer']

logger = logging.getLogger(__name__)


class NaiveQuantizer(Quantizer):
    """quantize weight to 8 bits
    """
    def __init__(self, model, config_list):
        super().__init__(model, config_list)
        self.layer_scale = {}

    def quantize_weight(self, weight, config, op_name, **kwargs):
        new_scale = weight.abs().max() / 127
        scale = max(self.layer_scale.get(op_name, 0), new_scale)
        self.layer_scale[op_name] = scale
        orig_type = weight.type()  # TODO: user layer
        return weight.div(scale).type(torch.int8).type(orig_type).mul(scale)

class EMA_RangeChecker:
    def __init__(self, alpha):
        self.alpha = alpha
        self.args = None

    def update(self, args):
        if self.args is None:
            if not isinstance(args, list):
                raise TypeError("expect type list of parameter args")
            self.args = args
        else:
            self.args = [self.alpha * args[idx] + (1 - self.alpha) * self.args[idx] for idx in range(len(self.args))]
        return self.args

class QAT_Quantizer(Quantizer):
    """Quantizer using the DoReFa scheme, as defined in:
    Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference
    http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf
    """
    def __init__(self, model, config_list):
        """
        Create a wrapper forward function to replace the original one.

        Parameters
        ----------
        layer : LayerInfo
            the layer to instrument the mask
        config_list : list of dict
            list of configurations for quantization
            supported keys for dict:
                - weight_quantization
                - weight_bits
                - activation_quantization
                - activation_bits
                - quant_delay
        """
        super().__init__(model, config_list)
        for layer, config in self.get_modules_to_compress():
            if config.get("weight_quantization", False):
                layer.module.register_buffer("zero_point", None)
                layer.module.register_buffer("scale", None)
            if config.get("activation_quantization", False):
                layer.module.register_buffer("range_checker", EMA_RangeChecker(0.9))
        self.steps = 0

    def fixed_range_check(self, tensor):
        """
        determine the max value and min value of quantization target.

        Parameters
        ----------
        tensor : Tensor
            the Tensor to be quantized
        
        Returns
        -------
        (min, max) : (float, float)
        """
        return torch.min(tensor), torch.max(tensor)

    # def EMA_range_check(self, op, tensor):
    #     return torch.min(tensor), torch.max(tensor)

    def update_quantization_param(self, bits, op, rmin, rmax):
        """
        update the `zero_point` and `scale` of op.

        Parameters
        ----------
        bits : int
            quantization bits length
        op : torch.nn.module
            target module
        rmin : float
            min value of real value
        rmax : float
            max value of real value
        """
        # extend the [min, max] interval to ensure that it contains 0.
        # Otherwise, we would not meet the requirement that 0 be an exactly
        # representable value.
        rmin = min(rmin, 0)
        rmax = max(rmax, 0)

        # the min and max quantized values, as floating-point values
        qmin = 0
        qmax = 1 << bits - 1

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

        op.scale = scale
        op.zero_point = nudged_zero_point

    def quantize(self, bits, op, real_val):
        """
        quantize real value.

        Parameters
        ----------
        bits : int
            quantization bits length
        op : torch.nn.module
            target module
        real_val : float
            real value to be quantized
        
        Returns
        -------
        quantized_val : float
        """
        transformed_val = op.zero_point + real_val / op.scale
        qmin = 0
        qmax = 1 << bits - 1
        clamped_val = torch.clamp(transformed_val, qmin, qmax)
        quantized_val = torch.round(clamped_val)
        return quantized_val

    def dequantize(self, op, quantized_val):
        """
        dequantize quantized value.

        Parameters
        ----------
        op : torch.nn.module
            target module
        quantized_val : float
            quantized_val value to be dequantized

        Returns
        -------
        real_val : float
        """
        real_val = op.scale * (quantized_val - op.zero_point)
        return real_val

    def quantize_weight(self, weight, config, op, **kwargs):
        if config['weight_bits'] <= 1:
            return weight
        # if config['quant_delay'] > self._steps:
        #     return weight
        rmin, rmax = self.fixed_range_check(weight)
        self.update_quantization_param(config['weight_bits'], op, rmin, rmax)
        out = self.quantize(config['weight_bits'], op, weight)
        out = self.dequantize(op, out)
        return out

    def quantize_activation(self, activation, config, op, **kwargs):
        if config['activation_bits'] <= 1:
            return activation
        # if config['quant_delay'] > self._steps:
        #     return activation
        # TODO activation dynamic set a, b
        rmin, rmax = self.fixed_range_check(activation)
        self.update_quantization_param(config['activation_bits'], op, rmin, rmax)
        out = self.quantize(config['activation_bits'], op, activation)
        out = self.dequantize(op, out)
        return out

    def fold_bn(self, config, **kwargs):
        # TODO simulate folded weight
        pass

    def step(self):
        """override compressor step method, update _step attribute, quantization only happens after certain number of steps
        """
        self.steps += 1


class DoReFaQuantizer(Quantizer):
    """Quantizer using the DoReFa scheme, as defined in:
    Zhou et al., DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients
    (https://arxiv.org/abs/1606.06160)
    """
    def __init__(self, model, config_list):
        """
        config_list: supported keys:
            - q_bits
        """
        super().__init__(model, config_list)

    def quantize_weight(self, weight, config, **kwargs):
        out = weight.tanh()
        out = out / (2 * out.abs().max()) + 0.5
        out = self.quantize(out, config['q_bits'])
        out = 2 * out -1
        return out

    def quantize(self, input_ri, q_bits):
        scale = pow(2, q_bits)-1
        output = torch.round(input_ri*scale)/scale
        return output
