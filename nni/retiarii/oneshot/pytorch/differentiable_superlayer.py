
import torch
import torch.nn as nn
import torch.nn.functional as F

from .superlayer import ToSample


class DifferentiableValueChoiceSuperLayer:
    """
    Layer that has at least one valuechoice in it param list. Basic functions such as getting max/min/sampled candidates are
    implemented in this class.

    Attributes
    ----------
    name : str
        the unique identifier of the module it replaced
    args : Dict[str, Any]
        the parameter list of the original module

    Parameters
    ----------
    module : nn.Module:
        module to be replaced
    module_name : str
        the unique identifier of `module`
    """

    def max_candidate(self, attr_name, default = None):
        attr = self.args.get(attr_name, default)
        if isinstance(attr, ToSample):
            return max(attr.candidates[f'{self.name}_{attr_name}'])
        return attr

    def min_candidate(self, attr_name, default = None):
        attr = self.args.get(attr_name, default)
        if isinstance(attr, ToSample):
            return min(attr.candidates[f'{self.name}_{attr_name}'])
        return attr

    def sampled_candidate(self, attr_name, default = None):
        attr = self.args.get(attr_name, default)
        if isinstance(attr, ToSample):
            return attr.sampled_candidate(f'{self.name}_{attr_name}')
        return attr
    
    def candidate_len(self, attr_name):
        if attr_name not in self.args:
            return 0
        attr = self.args[attr_name]
        if not isinstance(attr, ToSample):
            return 0
        return len(attr.candidates[f'{self.name}_{attr_name}'])
    
    def generate_alpha(self):
        self.alphas = {}
        for name, attr in self.args:
            if isinstance(attr, ToSample):
                self.alphas[name] = nn.Parameter(torch.randn(len(attr)) * 1e-3)
    
    def export_best_alpha(self):
        res = {}
        for name, alpha in self.alphas:
            res[name] = torch.argmax(alpha).item()
        return res
    
    @property
    def attr_alpha(self, attr_name):
        return self.alphas[attr_name]

    @property
    def alpha(self):
        return self.alphas


class DartsSuperConv2d(nn.Conv2d, DifferentiableValueChoiceSuperLayer):
    """
    Only ``kernel_size`` is supported.
    """
    def __init__(self, module, name):
        self.name = name
        self.args = module.trace_kwargs
        
        
        # compulsorty params
        max_in_channel = self.args['in_channels']
        max_out_channel = self.args['out_channels']
        # kernel_size may be an int or tuple, we turn it into a tuple for simplicity
        self.max_kernel_size = self.max_kernel_size_candidate()
        if not isinstance(self.max_kernel_size, tuple):
            self.max_kernel_size = (self.max_kernel_size, self.max_kernel_size)

        # optional params
        # stride, padding and dilation are not necessary for init funtion, since `Conv2d`` directly accessed them in `forward`,
        # which means we can set them just before calling Conv2d.forward
        min_groups = self.min_candidate('groups', 1)

        # non-valuechoice params
        bias = self.args.get('bias', False)
        padding_mode = self.args.get('padding_mode', 'zeros')
        device = self.args.get('device', None)
        dtype = self.args.get('dtype', None)

        super().__init__(self, max_in_channel, max_out_channel, self.max_kernel_size,
            groups = min_groups, bias = bias, padding_mode = padding_mode, device = device, dtype = dtype)
        
        self.generate_alpha()
    
    def forward(self, input):
        in_chn = self.sampled_candidate('in_channels')
        out_chn = self.sampled_candidate('out_channels')
        kernel_size = self.sampled_candidate('kernel_size')
        sampled_kernel_a, sampled_kernel_b = kernel_size \
            if isinstance(kernel_size, tuple) else kernel_size, kernel_size

        # Users are supposed to make sure that candidates with the same index match each other.
        # No need to figure if the following three attributes are tuples or not, since Conv2d will handel them.
        self.stride = self.sampled_candidate('stride', 1)
        self.padding = self.sampled_candidate('padding', 0)
        self.dilation = self.sampled_candidate('dilation', 1)

        # F.conv2d will handle `groups`, but we still need to slice weight tensor
        self.groups = self.sampled_candidate('groups', 1)

        # take the small kernel from the center and round it to floor(left top)
        # Example:
        #   max_kernel = 5*5, sampled_kernel = 3*3, then we take [1: 4]
        #   max_kernel = 5*5, sampled_kernel = 2*2, then we take [1: 3]
        #   □ □ □ □ □   □ □ □ □ □
        #   □ ■ ■ ■ □   □ ■ ■ □ □
        #   □ ■ ■ ■ □   □ ■ ■ □ □
        #   □ ■ ■ ■ □   □ □ □ □ □
        #   □ □ □ □ □   □ □ □ □ □
        max_kernel_a, max_kernel_b = self.max_kernel_size
        kernel_a_left, kernel_b_top = (max_kernel_a - sampled_kernel_a) // 2, (max_kernel_b - sampled_kernel_b) // 2
        weight = self.weight[:out_chn, :in_chn // self.groups,
            kernel_a_left : kernel_a_left + sampled_kernel_a,
            kernel_b_top : kernel_b_top + sampled_kernel_b]
        bias = self.bias[:out_chn] if self.bias is not None else None

        return self._conv_forward(input, weight, bias)
    
    
    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self):
        for name, p in super().named_parameters():
            if name == 'alpha':
                continue
            yield name, p

    def export(self):
        return self.export_best_alpha()

