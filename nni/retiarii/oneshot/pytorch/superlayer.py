
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from re import S
from turtle import forward
from click import option
from numpy import isin
import torch.nn as nn
import torch.nn.functional as F


class ToSample:
    """
    Base class for all xxxChoice to be sampled. Different attributes candidates are sampled at once.

    Attributes
    ----------
    label : str
        the identifier of all ToSample objects
    n_candidates : int
        the length of candidates
    sampled : int
        the index of the sampled candidate. should not be larger than `n_candidates`
    candidates : Dict[str, List[Any]]
        the candidates for different attributes

    Parameters
    ----------
    label : str
        the identifier of all ToSample objects
    n_candidates : int
        the length of candidates
    sampled : int
        the index of the sampled candidate. should not be larger than `n_candidates`
    """
    def __init__(self, label, n_candidates = 0, sampled = -1) -> None:
        self.label = label
        self.n_candidates = n_candidates
        self.candidates = {}
        self.sampled = sampled
    

    def sampled_candidate(self, attr_name):
        return self.candidates[attr_name][self.sampled]
    

    def add_candidates(self, attr, candidates = []):
        assert len(candidates) == self.n_candidates, 'For ValueChoice with the same label, the number of candidates should also be ' \
            f'the same. ValueChoice `{self.label}` expects candidatas with a length of {self.n_candidates}, but got ' \
            f'{len(candidates)} for `{attr}`.'
        self.candidates[attr] = candidates
        
    
    def __len__(self):
        return self.n_candidates
    


class ENASValueChoice(ToSample):
    def __init__(self, value_choice):
        super().__init__(value_choice.label, len(value_choice.candidates))
        self.n_chosen = 1


class RandomValueChoice(ToSample):
    def __init__(self, value_choice):
        super().__init__(value_choice.label, len(value_choice.candidates))


def sampled_candidate(attr, name):
    if isinstance(attr, ToSample):
        attr = attr.sampled_candidate(name)
    return attr


def max_candidate(attr, name):
    if isinstance(attr, ToSample):
        attr = max(attr.candidates[name])
    return attr

def min_candidate(attr, name):
    if isinstance(attr, ToSample):
        attr = min(attr.candidates[name])
    return attr

# SuperLinear(nn.Linear) 放一个单独文件
class PathSamplingSuperLinear(nn.Linear):
    def __init__(self, module, name) -> None:
        args = module.trace_kwargs
        self.name_in_parent = name

        self._in_features = args['in_features']
        max_in_features = max_candidate(self._in_features, f'{self.name_in_parent}_in_features')
        
        self._out_features = args['out_features']
        max_out_features = max_candidate(self._out_features, f'{self.name_in_parent}_out_features')

        bias = args.get('bias', True)
        device = args.get('device', None)
        dtype = args.get('dtype', None)
        
        super().__init__(max_in_features, max_out_features, bias, device, dtype)

    def forward(self, x):
        # 如果是 valuechoice 就去读 sample 的值，否则就是固定值
        in_dim = sampled_candidate(self._in_features, f'{self.name_in_parent}_in_features')
        out_dim = sampled_candidate(self._out_features, f'{self.name_in_parent}_out_features')
        
        weights = self.weight[:out_dim, :in_dim]
        bias = self.bias[:out_dim]
        
        return F.linear(x, weights, bias)

class PathSamplingSuperConv2d(nn.Conv2d):
    # 暂时只支持正方形的 kernel
    # 暂不支持 group
    # 也不支持嵌套
    def __init__(self, module, name):
        args = module.trace_kwargs
        self.name_in_parent = name
        
        # compulsorty params
        self._in_channels = args['in_channels']
        max_in_channel = max_candidate(self._in_channels, f'{self.name_in_parent}_in_channels')
        self._out_channels = args['out_channels']
        max_out_channel = max_candidate(self._out_channels, f'{self.name_in_parent}_out_channels')
        # kernel_size may be an int or tuple, we turn it into a tuple for simplicity
        self._kernel_size = args['kernel_size']
        self.max_kernel_size = self.max_kernel_size_candidate(self._kernel_size, f'{self.name_in_parent}_kernel_size')
        if not isinstance(self.max_kernel_size, tuple):
            self.max_kernel_size = (self.max_kernel_size, self.max_kernel_size)
        
        # optional params
        self._stride = args.get('stride', 1)
        self._padding = args.get('padding', 0)
        self._dilation = args.get('dilation', 1)
        self._groups = args.get('groups', 1)
        min_groups = min_candidate(self._groups, f'{self.name_in_parent}_groups')

        self._bias = args.get('bias', False)
        _padding_mode = args.get('padding_mode', 'zeros')
        _device = args.get('device', None)
        _dtype = args.get('dtype', None)
        super().__init__(max_in_channel, max_out_channel, self.max_kernel_size, groups = min_groups,
            padding_mode = _padding_mode, device = _device, dtype = _dtype)
    
    def forward(self, input):
        in_chn = sampled_candidate(self._in_channels, f'{self.name_in_parent}_in_channels')
        out_chn = sampled_candidate(self._out_channels, f'{self.name_in_parent}_out_channels')
        kernel_size = sampled_candidate(self._kernel_size, f'{self.name_in_parent}_kernel_size')
        sampled_kernel_a, sampled_kernel_b = kernel_size \
            if isinstance(kernel_size, tuple) else kernel_size, kernel_size

        # Users are supposed to make sure that candidates with the same index fit each other.
        # No need to figure if the following three attributes are tuples or not, since Conv2d already handeled it.
        self.stride = sampled_candidate(self._stride, f'{self.name_in_parent}_stride')
        self.padding = sampled_candidate(self._padding, f'{self.name_in_parent}_padding')
        self.dilation = sampled_candidate(self._dilation, f'{self.name_in_parent}_dilation')

        # F.conv2d will handle `groups`, but we still need to slice weight 
        self.groups = sampled_candidate(self._groups, f'{self.name_in_parent}_groups')

        # take the small kernel from the centre and round to floor(left top)
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
        bias = self.bias[:out_chn]

        return self._conv_forward(input, weight, bias)
    
    @staticmethod
    def max_kernel_size_candidate(kernel_size, name):
        if isinstance(kernel_size, ToSample):
            if isinstance(kernel_size.candidates[name][0], tuple):
                maxa, maxb = 0, 0
                for a, b in kernel_size.candidates[name]:
                    a = max(a, maxa)
                    b = max(b, maxb)
                kernel_size = (maxa, maxb)
            else:
                kernel_size = max_candidate(kernel_size, name)
        return kernel_size

class PathSamplingSuperBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, module, name):
        args = module.trace_kwargs
        self.name_in_parent = name
        # compulsory param
        self._num_features = args['num_features']
        max_num_features = max_candidate(self._num_features, f'{self.name_in_parent}_num_features')
        
        # the initial eps and momentum doesn't affect the behaviour of construction function
        # so min_candidate also works below 
        self._eps = args.get('eps', 1e-5)
        eps = max_candidate(self._eps, f'{self.name_in_parent}_eps')
        self._momentum = args.get('momentum', .1)
        momentum = max_candidate(self._momentum, f'{self.name_in_parent}_momentum')

        # no ValueChoice params
        affine = args.get('affine', True)
        track_running_stats = args.get('track_running_stats', True)
        device = args.get('device', None)
        dtype = args.get('dtype', None)

        super().__init__(max_num_features, eps, momentum, affine, track_running_stats, device, dtype)
    
    def forward(self, input):
        # get sampled parameters
        num_features = sampled_candidate(self._num_features, f'{self.name_in_parent}_num_features')
        weight = self.weight[:num_features]
        bias = self.bias[:num_features]
        running_mean = self.running_mean[:num_features]
        running_var = self.running_var[:num_features]

        self.eps = sampled_candidate(self._eps, f'{self.name_in_parent}_eps')
        self.momentum = sampled_candidate(self._momentum, f'{self.name_in_parent}_momentum')

        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            running_mean
            if not self.training or self.track_running_stats
            else None,
            running_var if not self.training or self.track_running_stats else None,
            weight,
            bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )



    