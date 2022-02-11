# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

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


class ValueChoiceSuperLayer:
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
    def __init__(self, module, module_name):
        self.name = module_name
        self.args = module.trace_kwargs

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


class ENASValueChoice(ToSample):
    def __init__(self, value_choice):
        super().__init__(value_choice.label, len(value_choice.candidates))
        self.n_chosen = 1


class RandomValueChoice(ToSample):
    def __init__(self, value_choice):
        super().__init__(value_choice.label, len(value_choice.candidates))


class PathSamplingSuperLinear(ValueChoiceSuperLayer, nn.Linear):
    """
    The Linear layer to replace original linear with valuechoices in its parameter list. It construct the biggest weight matrix first,
    and slice it before every forward according to the sampled value. Supported parameters are listed below:
        in_features : int
        out_features : int

    Parameters
    ----------
    module : nn.Module:
        module to be replaced
    name : str
        the unique identifier of `module`
    """
    def __init__(self, module, name) -> None:
        ValueChoiceSuperLayer.__init__(self, module, name)

        # compulsory params
        max_in_features = self.max_candidate('in_features')
        max_out_features = self.max_candidate('out_features')

        # optional and no valuechoice params
        bias = self.args.get('bias', True)
        device = self.args.get('device', None)
        dtype = self.args.get('dtype', None)

        nn.Linear.__init__(self, max_in_features, max_out_features, bias, device, dtype)

    def forward(self, x):
        in_dim = self.sampled_candidate('in_features')
        out_dim = self.sampled_candidate('out_features')

        weights = self.weight[:out_dim, :in_dim]
        bias = self.bias[:out_dim]

        return F.linear(x, weights, bias)


class PathSamplingSuperConv2d(ValueChoiceSuperLayer, nn.Conv2d):
    """
    The Conv2d layer to replace original conv2d with valuechoices in its parameter list. It construct the biggest weight matrix first,
    and slice it before every forward according to the sampled value.
    Supported valuechoice parameters are listed below:
        in_channels : int
        out_channels : int
        kernel_size : int, tuple(int)
        stride : int, tuple(int)
        padding : int, tuple(int)
        dilation : int, tuple(int)
        group : int
    
    Warnings
    ----------
    Users are supposed to make sure that in different valuechoices with the same label, candidates with the same index should match
    each other. For example, the constraint among `kernel_size`, `padding`, `stride` and `dilation` in a convolutional layer should
    be met. Users ought to design candidates carefully to produce a tensor with correct shape for downstream calculation.

    Parameters
    ----------
    module : nn.Module
        the module to be replaced
    name : str
        the unique identifier of `module`
    """
    def __init__(self, module, name):
        ValueChoiceSuperLayer.__init__(self, module, name)

        # compulsorty params
        max_in_channel = self.max_candidate('in_channels')
        max_out_channel = self.max_candidate('out_channels')
        # kernel_size may be an int or tuple, we turn it into a tuple for simplicity
        self.max_kernel_size = self.max_kernel_size_candidate()
        if not isinstance(self.max_kernel_size, tuple):
            self.max_kernel_size = (self.max_kernel_size, self.max_kernel_size)

        # optional params
        # stride, padding and dilation are not necessary for init funtion, since `Conv2d`` directly accessed them in `forward`,
        # which means we can set them just before calling Conv2d.forward
        min_groups = self.min_candidate('groups', 1)

        # no valuechoice params
        bias = self.args.get('bias', False)
        padding_mode = self.args.get('padding_mode', 'zeros')
        device = self.args.get('device', None)
        dtype = self.args.get('dtype', None)

        nn.Conv2d.__init__(self, max_in_channel, max_out_channel, self.max_kernel_size,
            groups = min_groups, bias = bias, padding_mode = padding_mode, device = device, dtype = dtype)

    def forward(self, input):
        in_chn = self.sampled_candidate('in_channels')
        out_chn = self.sampled_candidate('out_channels')
        kernel_size = self.sampled_candidate('kernel_size')
        sampled_kernel_a, sampled_kernel_b = kernel_size \
            if isinstance(kernel_size, tuple) else kernel_size, kernel_size

        # Users are supposed to make sure that candidates with the same index match each other.
        # No need to figure if the following three attributes are tuples or not, since Conv2d will handeled them.
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

    def max_kernel_size_candidate(self):
        kernel_size = self.args['kernel_size']

        if not isinstance(kernel_size, ToSample):
            return kernel_size

        candidates = kernel_size.candidates[f'{self.name}_kernel_size']
        if not isinstance(candidates[0], tuple):
            return max(candidates)

        maxa, maxb = 0, 0
        for a, b in candidates:
            a = max(a, maxa)
            b = max(b, maxb)
        return maxa, maxb


class PathSamplingSuperBatchNorm2d(ValueChoiceSuperLayer, nn.BatchNorm2d):
    """
    The BatchNorm2d layer to replace original bn2d with valuechoice in its parameter list. It construct the biggest mean and variation
    tensor first, and slice it before every forward according to the sampled value. Supported parameters are listed below:
        num_features : int
        eps : float
        momentum : float
    
    Parameters
    ----------
    module : nn.Module
        the module to be replaced
    name : str
        the unique identifier of `module`
    """
    def __init__(self, module, name):
        ValueChoiceSuperLayer.__init__(self, module, name)

        # compulsory params
        max_num_features = self.max_candidate('num_features')

        # optional params
        # the initial values of eps and momentum doesn't matter since they are directly accessed in forward
        # we just take max candidate for simplicity here
        eps = self.max_candidate('eps', 1e-4)
        momentum = self.max_candidate('momentum', .1)

        # no ValueChoice params
        affine = self.args.get('affine', True)
        track_running_stats = self.args.get('track_running_stats', True)
        device = self.args.get('device', None)
        dtype = self.args.get('dtype', None)

        nn.BatchNorm2d.__init__(self, max_num_features, eps, momentum, affine, track_running_stats, device, dtype)

    def forward(self, input):
        # get sampled parameters
        num_features = self.sampled_candidate('num_features')
        weight = self.weight[:num_features]
        bias = self.bias[:num_features]
        running_mean = self.running_mean[:num_features]
        running_var = self.running_var[:num_features]

        self.eps = self.sampled_candidate('eps', 1e-4)
        self.momentum = self.sampled_candidate('momentum', .1)

        # code below are simply coppied from pytorch v1.10.1 source code since directly setting weight or bias is not allowed.
        # please turn to pytorch source code if you have any problem with code below
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)
        # code above are simply coppied from pytorch v1.10.1 source code since directly setting weight or bias is not allowed.
        # please turn to pytorch source code if you have any problem with code above

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


class PathSamplingMultiHeadAttention(ValueChoiceSuperLayer, nn.MultiheadAttention):
    """
    The MultiHeadAttention layer to replace original mhattn with valuechoice in its parameter list. It construct the biggest Q, K,
    V and some other tensors first, and slice it before every forward according to the sampled value. Supported parameters are listed
    below:
        embed_dim : int
        num_heads : float
        kdim :int
        vdim : int
        dropout : float
    
    Warnings
    ----------
    Users are supposed to make sure that in different valuechoices with the same label, candidates with the same index should match
    each other. For example, the divisibility constraint between `embed_dim` and `num_heads` in a multi-head attention module should
    be met. Users ought to design candidates carefully to prevent the module from breakdown.
    
    Parameters
    ----------
    module : nn.Module
        the module to be replaced
    name : str
        the unique identifier of `module`
    """
    def __init__(self, module, name):
        ValueChoiceSuperLayer.__init__(self, name, module)

        # compulsory params
        self.max_embed_dim = self.max_candidate('embed_dim')
        num_heads = self.max_candidate('num_heads')
        
        # optional params
        kdim = self.max_candidate('kdim', self.max_embed_dim)
        vdim = self.max_candidate('vdim', self.max_embed_dim)
        dropout = self.max_candidate('dropout', 0.)

        # no valuechoice params
        bias = self.args.get('bias', True)
        add_bias_kv = self.args.get('add_bias_kv', False)
        add_zero_attn = self.args.get('add_zero_attn', False)
        batch_first = self.args.get('batch_first', False)
        device = self.args.get('device', None)
        dtype = self.args.get('dtype', None)

        nn.MultiheadAttention.__init__(self.max_embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn,
            kdim, vdim, batch_first ,device, dtype)
    
    def forward(self, query, key, value, key_padding_mask = None, need_weights = True, attn_mask = None):
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        embed_dim = self.sampled_candidate('embed_dim')
        kdim = self.sampled_candidate('kdim', embed_dim)
        vdim = self.sampled_candidate('vdim', embed_dim)
        num_heads = self.sampled_candidate('num_heads')

        in_proj_bias = self.in_proj_bias[:embed_dim] + \
                       self.in_proj_bias[self.max_embed_dim : self.max_embed_dim + embed_dim] + \
                       self.in_proj_bias[2 * self.max_embed_dim : 2 * self.max_embed_dim + embed_dim] \
                       if self.in_proj_bias is not None else None
        in_proj_weight = self.in_proj_weight[:embed_dim, :embed_dim] + \
                         self.in_proj_weight[self.max_embed_dim : self.max_embed_dim + embed_dim, :embed_dim] + \
                         self.in_proj_weight[2 * self.max_embed_dim : 2 * self.max_embed_dim + embed_dim, :embed_dim] \
                         if self.in_proj_weight is not None else None
        bias_k = self.bias_k[:,:,:embed_dim] if self.bias_k is not None else None
        bias_v = self.bias_v[:, :, :embed_dim] if self.bias_v is not None else None
        out_proj_weight = self.out_proj.weight[:embed_dim, embed_dim]
        out_proj_bias = self.out_proj.bias[:embed_dim]

        if not self._qkv_same_embed_dim:
            q_proj = self.q_proj_weight[:embed_dim, :embed_dim]
            k_proj = self.k_proj_weight[:embed_dim, :kdim]
            v_proj = self.v_proj_weight[:embed_dim, :vdim]

            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, embed_dim, num_heads,
                in_proj_weight, in_proj_bias,
                bias_k, bias_v, self.add_zero_attn,
                self.dropout, out_proj_weight, out_proj_bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=q_proj, k_proj_weight=k_proj, v_proj_weight=v_proj)
        else:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, embed_dim, num_heads,
                in_proj_weight, in_proj_bias,
                bias_k, bias_v, self.add_zero_attn,
                self.dropout, out_proj_weight, out_proj_bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights
