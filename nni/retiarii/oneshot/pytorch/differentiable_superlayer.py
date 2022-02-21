import torch
import torch.nn as nn
import torch.nn.functional as F

from nni.retiarii.nn.pytorch.api import ValueChoice


class DifferentiableSuperConv2d(nn.Conv2d, object):
    """
    Only ``kernel_size`` ``in_channels`` and ``out_channels`` are supported. Kernel size candidates should be larger or smaller
    than each other in both candidates. See examples below:
    the following example is not allowed:
        >>> ValueChoice(candidates = [(5, 3), (3, 5)])
            □ ■ ■ ■ □   □ □ □ □ □
            □ ■ ■ ■ □   ■ ■ ■ ■ ■    # candidates are not strictly bigger or smaller
            □ ■ ■ ■ □   ■ ■ ■ ■ ■
            □ ■ ■ ■ □   ■ ■ ■ ■ ■
            □ ■ ■ ■ □   □ □ □ □ □
    the following 3 examples are valid:
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
        max_in_channel = self.args['in_channels']
        if isinstance(max_in_channel, ValueChoice):
            max_in_channel = max(max_in_channel.candidates)

        max_out_channel  = self.args['out_channels']
        self.out_channel_candidates = None
        if isinstance(max_out_channel, ValueChoice):
            self.out_channel_candidates = sorted(max_out_channel.candidates)
            max_out_channel =  self.out_channel_candidates[-1]

        # kernel_size may be an int or tuple, we turn it into a tuple for simplicity
        self.max_kernel_size = self.pre_process_kernel_size()

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

        super().__init__(max_in_channel, max_out_channel, self.max_kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, device, dtype)

        self.generate_architecture_params()

    def forward(self, input):
        # Note that there is no need to handle ``in_channels`` here since it is already handle by the ``out_channels`` in the
        # previous module. If we multiply alpha with refer to ``in_channels`` here again, the alpha will indeed be considered
        # twice, which is not what we expect.

        weight = self.weight

        # Note that ``self.xxx_mask`` and ``self.t_xxx`` have different lengths. There alignment is shown below:
        # self.xxx_candidates = [  k0   ,   k1   , ... ,  k_n-2  ,   k_n-1 ] # ascending order
        # self.xxx_mask       = [mask_0 , mask_1 , ... , mask_n-2, mask_n-1] # mask_n-1 is a matrix of ones
        # self.t_xxx          =     1    [ t_1   , ... ,  t_n-2  ,   t_n-1 ]
        # So we multiply weight with ``self.mask[0]`` at the very beginning, and zip the rest part.
        
        if self.kernel_size_candidates is not None:
            weight = weight * self.kernel_masks[0]
            # Note that we need the smaller_shape, or the shape in the previous iteration here, so we use candidates[:-1] rather
            # than candidates[1:]
            for mask, t, smaller_shape in zip(self.kernel_masks[1:], self.t_kernel, self.kernel_size_candidates[:-1]):
                bigger_weight = self.weight * mask
                # although the ``bigger_weight`` is of the same shape as ``processed_weight``, the outer line of it is zero-outed
                alpha = self.sigmoid_Lasso_kernel(bigger_weight, t, smaller_shape)
                weight *= 1 - alpha
                weight += alpha * bigger_weight

        # same as kernel_size candidates except Lasso function
        if self.out_channel_candidates is not None:
            processed_weight = weight
            weight = processed_weight * self.channel_masks[0]
            for mask, t, smaller_shape in zip(self.channel_masks[1:], self.t_expansion, self.out_channel_candidates[:-1]):
                bigger_weight = processed_weight * mask
                alpha = self.sigmoid_Lasso_channel(bigger_weight, t, smaller_shape)
                weight *= 1 - alpha
                weight += alpha * bigger_weight

        output = self._conv_forward(input ,weight, self.bias)
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
        inner_size : int
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
        result = {}
        for k, v in self.alpha.items():
            result[k] = torch.argsort(v).cpu().numpy().tolist()[0]
        return result

    def pre_process_kernel_size(self):
        """
        Unify kernel size candidates as tuples and sort them in ascending order if kernel size is a ``ValueChoice``.
        """
        kernel_size = self.args['kernel_size']
        self.kernel_size_candidates = None

        if not isinstance(kernel_size, ValueChoice):
            return kernel_size

        # unify kernel size as tuple
        candidates = kernel_size.candidates
        if not isinstance(candidates[0], tuple):
            for i in range(0, len(candidates)):
                candidates[i] = (candidates[i], candidates[i])

        # sort kernel size in ascending order
        self.kernel_size_candidates = sorted(candidates, key=lambda t : t[0])
        for i in range(1, len(self.kernel_size_candidates)):
            pre = self.kernel_size_candidates[i-1]
            cur = self.kernel_size_candidates[i]
            assert pre[1] <= cur[1], f'Kernel_size candidates should be larger or smaller than each other on both dimensions, but' \
                f' found {pre} and {cur}.'

        return self.kernel_size_candidates[-1]


    def generate_architecture_params(self):
        self.alpha = {}
        if self.kernel_size_candidates is not None:
            # kernel size arch params
            self.t_kernel = nn.Parameter(torch.randn(len(self.kernel_size_candidates) - 1))
            self.alpha['kernel_size'] = self.t_kernel
            # kernel size mask
            self.kernel_masks = []
            for kernel_size in self.kernel_size_candidates:
                mask = torch.zeros_like(self.weight)
                mask[:,:,:kernel_size[0], :kernel_size[1]] = 1
                self.kernel_masks.append(mask)

        if self.out_channel_candidates is not None:
            # expansion arch params
            self.t_expansion = nn.Parameter(torch.randn(len(self.out_channel_candidates) - 1))
            self.alpha['out_channels'] = self.t_expansion
            self.channel_masks = []
            for out_channel in self.out_channel_candidates:
                mask = torch.zeros_like(self.weight)
                mask[:out_channel] = 1
                self.channel_masks.append(mask)
