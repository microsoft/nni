import logging
import torch
from .compressor import Quantizer

__all__ = [ 'NaiveQuantizer', 'QAT_Quantizer', 'DoReFaQuantizer' ]

logger = logging.getLogger(__name__)


class NaiveQuantizer(Quantizer):
    """
    quantize weight to 8 bits
    """
    def __init__(self, config_list):
        super().__init__(config_list)
        self.layer_scale = {}

    def quantize_weight(self, weight, config, op_name, **kwargs):
        new_scale = weight.abs().max() / 127
        scale = max(self.layer_scale.get(op_name, 0), new_scale)
        self.layer_scale[op_name] = scale
        orig_type = weight.type()  # TODO: user layer
        return weight.div(scale).type(torch.int8).type(orig_type).mul(scale)


class QAT_Quantizer(Quantizer):
    """
    Quantizer using the DoReFa scheme, as defined in:
    Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference
    http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf
    """
    def __init__(self, config_list):
        """
            Configure Args:
                q_bits
        """
        super().__init__(config_list)

    def quantize_weight(self, weight, config, **kwargs):
        if config['q_bits'] <= 1:
            return weight
        a = torch.min(weight)
        b = torch.max(weight)
        n = pow(2, config['q_bits'])
        scale = (b-a)/(n-1)
        zero_point = a
        out = torch.round((weight - zero_point)/scale)
        out = out*scale + zero_point
        orig_type = weight.dtype
        return out.type(orig_type)


class DoReFaQuantizer(Quantizer):
    """
    Quantizer using the DoReFa scheme, as defined in:
    Zhou et al., DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients
    (https://arxiv.org/abs/1606.06160)
    """
    def __init__(self, config_list):
        """
            configure Args:
                q_bits
        """
        super().__init__(config_list)

    def quantize_weight(self, weight, config, **kwargs):
        out = weight.tanh()
        out = out /( 2 * out.abs().max()) + 0.5
        out = self.quantize(out, config['q_bits'])
        out = 2 * out -1
        return out

    def quantize(self, input_ri, q_bits):
        scale = pow(2, q_bits)-1
        output = torch.round(input_ri*scale)/scale
        return output
