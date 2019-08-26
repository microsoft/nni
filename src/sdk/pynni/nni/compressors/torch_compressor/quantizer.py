import torch
from ._nnimc_torch import TorchQuantizer
from ._nnimc_torch import _torch_default_get_configure, _torch_default_load_configure_file

class NaiveQuantizer(TorchQuantizer):
    def __init__(self):
        super().__init__()
        self.layer_scale = {}

    def quantize_weight(self, layer_info, weight):
        new_scale = weight.abs().max() / 127
        # TODO: use int id
        scale = max(self.layer_scale.get(layer_info.name, 0), new_scale)
        self.layer_scale[layer_info.name] = scale
        orig_type = weight.type()  # TODO: user layer_info
        return weight.div(scale).type(torch.int8).type(orig_type).mul(scale)

class DoReFaQuantizer(TorchQuantizer):
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
    
    def load_configure(self, config_path):
        config_list = _torch_default_load_configure_file(config_path, 'DoReFaQuantizer')
        for config in config_list.get('config', []):
            self.configure_list.append(config)
        
    def get_qbits(self, configure):
        if not isinstance(configure, dict):
            print('WARNING: you should input a dict to get_qbits, set DEFAULT { }')
            configure = {}
        qbits = configure.get('q_bits', 32)
        if qbits == 0:
            print('WARNING: you can not set q_bits ZERO!')
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
    
    def load_configure(self, config_path):
        config_list = _torch_default_load_configure_file(config_path, 'QATquantizer')
        for config in config_list.get('config', []):
            self.configure_list.append(config)
        
    def get_qbits(self, configure):
        if not isinstance(configure, dict):
            print('WARNING: you should input a dict to get_qbits, set DEFAULT { }')
            configure = {}
        qbits = configure.get('q_bits', 32)
        if qbits == 0:
            print('WARNING: you can not set q_bits ZERO!')
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
