from rsa import newkeys
import torch
import torch.nn as nn
import torch.nn.functional as F

from nni.retiarii.nn.pytorch.api import ValueChoice

from .superlayer import ToSample


class DifferentiableSuperConv2d(nn.Conv2d):
    """
    Only ``kernel_size`` is supported now. Kernel size must be odd numbers, and candidates should be strictly bigger or smaller
    than each other. See examples below:
    the following example is not allowed:
        >>> ValueChoice(candidates = [(5, 3), (3, 5)])
           □ ■ ■ ■ □   □ □ □ □ □
           □ ■ ■ ■ □   ■ ■ ■ ■ ■    # candidates are not strictly bigger or smaller
           □ ■ ■ ■ □   ■ ■ ■ ■ ■
           □ ■ ■ ■ □   ■ ■ ■ ■ ■
           □ ■ ■ ■ □   □ □ □ □ □
        >>> ValueChoice(candidates = [(5, 5), (3, 5)])
           ■ ■ ■ ■ ■   □ □ □ □ □
           ■ ■ ■ ■ ■   ■ ■ ■ ■ ■
           ■ ■ ■ ■ ■   ■ ■ ■ ■ ■    # candidates are not strictly bigger or smaller. the second dimension values are equal
           ■ ■ ■ ■ ■   ■ ■ ■ ■ ■
           ■ ■ ■ ■ ■   □ □ □ □ □
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
    """
    def __init__(self, module, name):
        self.label = name
        self.args = module.trace_kwargs
        
        # compulsory params
        in_channel = self.args['in_channels']
        out_channel = self.args['out_channels']
        # kernel_size may be an int or tuple, we turn it into a tuple for simplicity
        self.max_kernel_size = self.validate_kernel_size()

        # optional params
        stride = self.args.get('stride', 1)
        padding = self.args.get('padding', 0)
        dilation = self.args.get('dilation', 1)

        min_groups = self.args.get('groups', 1)

        # non-valuechoice params
        bias = self.args.get('bias', False)
        padding_mode = self.args.get('padding_mode', 'zeros')
        device = self.args.get('device', None)
        dtype = self.args.get('dtype', None)

        super().__init__(in_channel, out_channel, self.max_kernel_size, stride, padding, dilation,
            min_groups, bias, padding_mode, device, dtype)
        
        self.generate_t()
    
    def forward(self, input):
        # calculate new kernel
        weight = self.calculate_kernel_weight()

        # F.conv2d will handle `groups`, but we still need to slice weight tensor
        self.groups = self.args.get('groups', 1)

        return self._conv_forward(input, weight, self.bias)
    
    @staticmethod
    def sigmoid_Lasso(matrix, inner_size, t):
        """
        calculate differentiable sigmoid lasso term. see the paper for details.
        Parameters
        ----------
        matrix : Tensor[]
            the big matrix to calculate lasso norm, no need to zero-out inner weights
        inner_size : int, tuple
            the inner kernel size to be gotten rid of, ignore in_channel out_channel
        t : threshold
            the threshold
        """
        if not isinstance(inner_size, tuple):
            inner_size = (inner_size, inner_size)
        left = (matrix.shape[-2] - inner_size[0]) // 2
        top = (matrix.shape[-1] - inner_size[1]) // 2
        lasso = torch.norm(matrix) - torch.norm(matrix[:, :, left : left + inner_size[0], top : top + inner_size[1]])
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
        return self.t_kernel

    def generate_t(self):
        if self.kernel_size_candidates is not None:
            self.t_kernel = nn.Parameter(torch.randn(len(self.kernel_size_candidates) - 1))
    
    def calculate_kernel_weight(self):
        w, h = self.weight.shape[-2], self.weight.shape[-1]
        
        new_kernel = torch.zeros_like(self.weight)
        # kernel sizes are in ascending order
        for k, (a, b) in enumerate(self.kernel_size_candidates):
            l, t = (w - a) // 2, (h - b) // 2
            indicator = 0 
            if k > 0:
                indicator = self.sigmoid_Lasso(self.weight[:, :, l : l + a, t : t + b],
                    self.kernel_size_candidates[k - 1], self.t_kernel[k - 1])
            new_kernel = new_kernel * indicator
            for out_c in range(0, self.out_channels):
                for in_c in range(0, self.in_channels):
                    for i in range(l, l + a):
                        for j in range(t, t + b):
                            new_kernel[out_c][in_c][i][j] += (1 - indicator) * self.weight[out_c][in_c][i][j]

        return new_kernel

            
        

