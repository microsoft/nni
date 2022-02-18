from numpy import isin
from rsa import newkeys
import torch
import torch.nn as nn
import torch.nn.functional as F

from nni.retiarii.nn.pytorch.api import ValueChoice

from .superlayer import ToSample


class DifferentiableSuperConv2d(nn.Conv2d):
    """
    Only ``kernel_size`` is supported now. Candidates should be strictly bigger or smaller than each other (or, for both dimension).
    See examples below:
    the following example is not allowed:
        >>> ValueChoice(candidates = [(5, 3), (3, 5)])
            □ ■ ■ ■ □   □ □ □ □ □
            □ ■ ■ ■ □   ■ ■ ■ ■ ■    # candidates are not strictly bigger or smaller
            □ ■ ■ ■ □   ■ ■ ■ ■ ■
            □ ■ ■ ■ □   ■ ■ ■ ■ ■
            □ ■ ■ ■ □   □ □ □ □ □
    the following examples are valid:
        >>> ValueChoice(candidates = [5, 3, 1])
            ■ ■ ■ ■ ■   □ □ □ □ □   □ □ □ □ □
            ■ ■ ■ ■ ■   □ ■ ■ ■ □   □ □ □ □ □
            ■ ■ ■ ■ ■   □ ■ ■ ■ □   □ □ ■ □ □
            ■ ■ ■ ■ ■   □ ■ ■ ■ □   □ □ □ □ □
            ■ ■ ■ ■ ■   □ □ □ □ □   □ □ □ □ □
        >>> ValueChoice(candidates = [(5, 7), (3, 5), (1, 3)])
            ■ ■ ■ ■ ■ ■ ■  □ □ □ □ □ □ □   □ □ □ □ □ □ □
            ■ ■ ■ ■ ■ ■ ■  □ ■ ■ ■ ■ ■ □   □ □ □ □ □ □ □
            ■ ■ ■ ■ ■ ■ ■  □ ■ ■ ■ ■ ■ □   □ □ ■ ■ ■ □ □
            ■ ■ ■ ■ ■ ■ ■  □ ■ ■ ■ ■ ■ □   □ □ □ □ □ □ □
            ■ ■ ■ ■ ■ ■ ■  □ □ □ □ □ □ □   □ □ □ □ □ □ □
        >>> # when the difference between any two candidates is not even, the left upper will be picked:
        >>> ValueChoice(candidates = [(5, 5), (4, 4), (3, 3)])
            ■ ■ ■ ■ ■   ■ ■ ■ ■ □   □ □ □ □ □
            ■ ■ ■ ■ ■   ■ ■ ■ ■ □   □ ■ ■ ■ □
            ■ ■ ■ ■ ■   ■ ■ ■ ■ □   □ ■ ■ ■ □
            ■ ■ ■ ■ ■   ■ ■ ■ ■ □   □ ■ ■ ■ □
            ■ ■ ■ ■ ■   □ □ □ □ □   □ □ □ □ □
    """
    def __init__(self, module, name):
        self.label = name
        self.args = module.trace_kwargs
        
        # compulsory params
        in_channel = self.args['in_channels']
        max_out_channel = self.out_channel_candidates = self.args['out_channels']
        if isinstance(self.out_channel_candidates, ValueChoice):
            self.out_channel_candidates.candidates = sorted(self.out_channel_candidates.candidates)
            max_out_channel =  self.out_channel_candidates[-1]
        # kernel_size may be an int or tuple, we turn it into a tuple for simplicity
        self.max_kernel_size = self.validate_kernel_size()

        # optional params
        stride = self.args.get('stride', 1)
        padding = self.args.get('padding', 0)
        dilation = self.args.get('dilation', 1)

        groups = self.args.get('groups', 1)

        # non-valuechoice params
        bias = self.args.get('bias', False)
        padding_mode = self.args.get('padding_mode', 'zeros')
        device = self.args.get('device', None)
        dtype = self.args.get('dtype', None)

        super().__init__(in_channel, max_out_channel, self.max_kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, device, dtype)
        
        self.generate_architecture_params()
    
    def forward(self, input):
        # self.mask =     [mask_0 , mask_1 , ... , mask_n-2]   1  
        # self.t_kernel =     1    [ t_1   , ... ,  t_n-2   , t_n-1]
        if self.kernel_size_candidates is not None:
            weight = self.weight * self.mask[0]
            output = self._conv_forward(input, weight, self.bias)

            for mask, t_k, candidate_shape in zip(self.mask[1:], self.t_kernel[: -1], self.kernel_size_candidates[1: -1]):
                weight = self.weight * mask
                l = (self.weight.shape[2] - candidate_shape[0]) // 2
                t = (self.weight.shape[3] - candidate_shape[1]) // 2
                alpha = self.sigmoid_Lasso_kernel(
                    self.weight[:, :, l : l + candidate_shape[0], t : t + candidate_shape[1]],
                    t_k, candidate_shape)
                output *= 1 - alpha
                output += alpha * self.self._conv_forward(input, weight, self.bias)
            
            alpha = self.sigmoid_Lasso_kernel(self.weight, self.t_kernel[-1], self.kernel_size_candidates[-1])
            output *= 1 - alpha
            output += alpha * self._conv_forward(input, self.weight, self.bias)
        else:
            output = self._conv_forward(input, self.weight, self.bias)

        if isinstance(self.out_channel_candidates, list):
            mask = torch.zeros_like(output)
            mask[:self.out_channel_candidates[0]] = 1
            full_channel_output = output
            output = full_channel_output * mask
            for small_cha, big_cha, t_expansion in zip(self.out_channel_candidates[:-1], self.out_channel_candidates[1:], self.t_expansion):
                mask[:big_cha] = 1
                alpha = self.sigmoid_Lasso_channel(self.weight[:big_cha], small_cha, t_expansion)
                output *= 1 - alpha
                output += full_channel_output * mask * alpha
        
        return output
    
    @staticmethod
    def sigmoid_Lasso_kernel(matrix, t, inner_size):
        """
        calculate differentiable sigmoid lasso term for kernel size. see the paper for details.
        Parameters
        ----------
        matrix : Tensor[]
            the big matrix to calculate lasso norm, no need to zero-out inner weights
        t : threshold
            the threshold
        inner_size : int, tuple
            the inner kernel size to be gotten rid of, ignore in_channel out_channel
        """
        if not isinstance(inner_size, tuple):
            inner_size = (inner_size, inner_size)
        left = (matrix.shape[-2] - inner_size[0]) // 2
        top = (matrix.shape[-1] - inner_size[1]) // 2
        lasso = torch.norm(matrix) - torch.norm(matrix[:, :, left : left + inner_size[0], top : top + inner_size[1]])
        sig = F.sigmoid(torch.square(lasso) - t)
        return sig

    @staticmethod
    def sigmoid_Lasso_channel(matrix, t, inner_size):
        """
        calculate differentiable sigmoid lasso term for expansion ratio. see the paper for details.
        Parameters
        ----------
        matrix : Tensor[]
            the big matrix to calculate lasso norm, no need to zero-out inner weights
        t : threshold
            the threshold
        inner_size : int, tuple
            the inner out_channel count to be gotten rid of
        """
        lasso = torch.norm(matrix) - torch.norm(matrix[:, :inner_size])
        sig = F.sigmoid(torch.square(lasso) - t)
        return sig
    
    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self):
        for name, p in super().named_parameters():
            if name == 'alpha':
                continue
            yield name, p

    def export(self):
        return torch.argsort(self.alpha).cpu().numpy().tolist()[0]
    
    def validate_kernel_size(self):
        kernel_size = self.args['kernel_size']

        if not isinstance(kernel_size, ValueChoice):
            return kernel_size
        
        # unify kernel size as tuple
        candidates = kernel_size.candidates
        if not isinstance(candidates[0], tuple):
            for i in range(0, len(candidates)):
                candidates[i] = (candidates[i], candidates[i])
        
        # sort kernel size in ascending order    
        for k, v in candidates:
            assert k % 2 == 1 and v % 2 == 1, 'kernel size must be odd numbers in valuechoice!'
        self.kernel_size_candidates = sorted(candidates, key=lambda t : t[0])
        return self.kernel_size_candidates[-1]
    
    @property
    def alpha(self):
        return [self.t_kernel, self.t_expansion]

    def generate_architecture_params(self):
        if self.kernel_size_candidates is not None:    
            # kernel size arch params 
            self.t_kernel = nn.Parameter(torch.randn(len(self.kernel_size_candidates) - 1))
            
            # kernel size mask
            self.mask = []
            for kernel_size in self.kernel_size_candidates[1:]:
                mask = torch.zeros_like(self.weight)
                mask[:,:,kernel_size[0], kernel_size[1]] = 1
                self.mask.append(mask)

        out_channel_candidates = self.args['out_channel']
        if isinstance(out_channel_candidates, ValueChoice):
            # expansion arch params
            self.t_expansion = nn.parameter(torch.randn(len(out_channel_candidates) - 1))