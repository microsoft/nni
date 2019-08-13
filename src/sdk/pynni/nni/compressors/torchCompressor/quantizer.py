try:
    import torch
    from ._nnimc_torch import TorchQuantizer
    
    class TorchNaiveQuantizer(TorchQuantizer):
        def __init__(self):
            super().__init__()
            self.layer_scale = { }
    
        def quantize_weight(self, layer_info, weight):
            new_scale = weight.abs().max() / 127
            # TODO: use int id
            scale = max(self.layer_scale.get(layer_info.name, 0), new_scale)
            self.layer_scale[layer_info.name] = scale
            orig_type = weight.type()  # TODO: user layer_info
            return weight.div(scale).type(torch.int8).type(orig_type).mul(scale)
    
    class TorchDoReFaQuantizer(TorchQuantizer):
        def __init__(self, q_bits):
            super().__init__()
            self.q_bits = q_bits
        
        def quantize_weight(self, layer_info, weight):
            out = weight.tanh()
            out = out /( 2 * out.abs().max()) + 0.5
            out = self.quantize(out, self.q_bits)
            out = 2 * out -1
            return out
        
        def quantize(self, input_ri, q_bits):
            scale = pow(2, q_bits)-1
            output = torch.round(input_ri*scale)/scale
            return output

    class TorchQATquantizer(TorchQuantizer):
        def __init__(self, q_bits):
            super().__init__()
            self.q_bits = q_bits
        
        def quantize_weight(self, layer_info, weight):
            if self.q_bits <= 1:
                return weight
            a = torch.min(weight)
            b = torch.max(weight)
            n = pow(2,self.q_bits)
            scale = (b-a)/(n-1)
            zero_point = a
            out = torch.round((weight - zero_point)/scale)
            out = out*scale + zero_point
            orig_type = weight.dtype
            return out.type(orig_type)

except ModuleNotFoundError:
    pass
