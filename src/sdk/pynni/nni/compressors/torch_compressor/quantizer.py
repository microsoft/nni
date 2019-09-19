import torch
import torch.nn as nn
from ._nnimc_torch import TorchQuantizer
from ._nnimc_torch import _torch_default_get_configure, _torch_default_load_configure_file, _torch_detect_module
from .Conv2d_KSE import Conv2d_KSE
import logging
logger = logging.getLogger('torch quantizer')


class NaiveQuantizer(TorchQuantizer):
    """
    quantize weight to 8 bits
    """
    def __init__(self):
        super().__init__()
        self.layer_scale = {}

    def quantize_weight(self, layer_info, weight):
        new_scale = weight.abs().max() / 127
        scale = max(self.layer_scale.get(layer_info.name, 0), new_scale)
        self.layer_scale[layer_info.name] = scale
        orig_type = weight.type()  # TODO: user layer_info
        return weight.div(scale).type(torch.int8).type(orig_type).mul(scale)

class DoReFaQuantizer(TorchQuantizer):
    """
    Quantizer using the DoReFa scheme, as defined in:
    Zhou et al., DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients
    (https://arxiv.org/abs/1606.06160)
    """
    def __init__(self, configure_list):
        """
            configure Args:
                q_bits
        """
        super().__init__()
        self.configure_list = []
        if isinstance(configure_list, list):
            for configure in configure_list:
                self.configure_list.append(configure)
        else:
            raise ValueError('please init with configure list')
    
        
    def get_qbits(self, configure):
        if not isinstance(configure, dict):
            logger.warning('WARNING: you should input a dict to get_qbits, set DEFAULT { }')
            configure = {}
        qbits = configure.get('q_bits', 32)
        if qbits == 0:
            logger.warning('WARNING: you can not set q_bits ZERO!')
            qbits = 32
        return qbits

    def quantize_weight(self, layer_info, weight):
        q_bits = self.get_qbits(_torch_default_get_configure(self.configure_list, layer_info))

        out = weight.tanh()
        out = out /( 2 * out.abs().max()) + 0.5
        out = self.quantize(out, q_bits)
        out = 2 * out -1
        return out
    
    def quantize(self, input_ri, q_bits):
        scale = pow(2, q_bits)-1
        output = torch.round(input_ri*scale)/scale
        return output

class QATquantizer(TorchQuantizer):
    """
    Quantizer using the DoReFa scheme, as defined in:
    Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference
    http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf
    """
    def __init__(self, configure_list):
        """
            Configure Args:
                q_bits
        """
        super().__init__()
        self.configure_list = []
        if isinstance(configure_list, list):
            for configure in configure_list:
                self.configure_list.append(configure)
        else:
            raise ValueError('please init with configure list')
    
        
    def get_qbits(self, configure):
        if not isinstance(configure, dict):
            logger.warning('WARNING: you should input a dict to get_qbits, set DEFAULT { }')
            configure = {}
        qbits = configure.get('q_bits', 32)
        if qbits == 0:
            logger.warning('WARNING: you can not set q_bits ZERO!')
            qbits = 32
        return qbits

    def quantize_weight(self, layer_info, weight):
        q_bits = self.get_qbits(_torch_default_get_configure(self.configure_list, layer_info))

        if q_bits <= 1:
            return weight
        a = torch.min(weight)
        b = torch.max(weight)
        n = pow(2, q_bits)
        scale = (b-a)/(n-1)
        zero_point = a
        out = torch.round((weight - zero_point)/scale)
        out = out*scale + zero_point
        orig_type = weight.dtype
        return out.type(orig_type)


class KSE(TorchQuantizer):
    """
    Use algorithm from "ExploitingKernelSparsityandEntropyforInterpretableCNNCompression" 
    https://arxiv.org/abs/1812.04368
    """
    def __init__(self, configure_list):
        """
            configure Args:
                G: 
                T: 
        """
        super().__init__()
        self.configure_list = []
        if isinstance(configure_list, list):
            for configure in configure_list:
                self.configure_list.append(configure)
        else:
            raise ValueError('please init with configure list')

    def get_GT(self, configure):
        if not isinstance(configure, dict):
            logger.warning('WARNING: you should input a dict to get_GT, set DEFAULT { }')
            configure = {}
        G = configure.get('G', 4)
        T = configure.get('T', 0)
        return G, T
    
    def compress(self, model):
        super().compress(model)
        for layer_info in _torch_detect_module(model, nn.Conv2d):
            G, T = self.get_GT(_torch_default_get_configure(self.configure_list, layer_info))

            # replace origin conv2d with conv2d_kse
            conv2d_kse = Conv2d_KSE(
                input_channels = layer_info.layer.in_channels, 
                output_channels = layer_info.layer.out_channels, 
                kernel_size = layer_info.layer.kernel_size, 
                stride = layer_info.layer.stride, 
                padding = layer_info.layer.padding, 
                bias=False, 
                G=G, 
                T=T)
            conv2d_kse.weight = layer_info.layer.weight
            conv2d_kse.bias = layer_info.layer.bias

            # calculate clusters and index
            conv2d_kse.KSE(G, T)
            conv2d_kse.forward_init()
            layer_info.layer = conv2d_kse
            setattr(model, layer_info.name, layer_info.layer)
            self._instrument_layer(layer_info)
    
    def _instrument_layer(self, layer_info):
        assert layer_info._forward is None
        layer_info._forward = layer_info.layer.forward

        def new_forward(*input):
            # layer_info.layer.weight.data = self.quantize_weight(layer_info, layer_info.layer.weight.data)
            return layer_info._forward(*input)

        layer_info.layer.forward = new_forward