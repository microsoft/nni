# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# pylint: skip-file
# type: ignore

"""This file is an incomplete implementation of `Single-path NAS <https://arxiv.org/abs/1904.02877>`__.
These are merely some components of the algorithm. The complete support is an undergoing work item.

Keep this file here so that it can be "blamed".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nni.retiarii.nn.pytorch import ValueChoice


class DifferentiableSuperConv2d(nn.Conv2d):
    """
    Only ``kernel_size`` ``in_channels`` and ``out_channels`` are supported. Kernel size candidates should be larger or smaller
    than each other in both candidates. See examples below:
    the following example is not allowed:
        >>> ValueChoice(candidates = [(5, 3), (3, 5)])
            □ ■ ■ ■ □   □ □ □ □ □
            □ ■ ■ ■ □   ■ ■ ■ ■ ■    # candidates are not bigger or smaller on both dimension
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
        args = module.trace_kwargs

        # compulsory params
        if isinstance(args['in_channels'], ValueChoice):
            args['in_channels'] = max(args['in_channels'].candidates)

        self.out_channel_candidates = None
        if isinstance(args['out_channels'], ValueChoice):
            self.out_channel_candidates = sorted(args['out_channels'].candidates, reverse=True)
            args['out_channels'] = self.out_channel_candidates[0]

        # kernel_size may be an int or tuple, we turn it into a tuple for simplicity
        self.kernel_size_candidates = None
        if isinstance(args['kernel_size'], ValueChoice):
            # unify kernel size as tuple
            candidates = args['kernel_size'].candidates
            if not isinstance(candidates[0], tuple):
                candidates = [(k, k) for k in candidates]

            # sort kernel size in descending order
            self.kernel_size_candidates = sorted(candidates, key=lambda t: t[0], reverse=True)
            for i in range(0, len(self.kernel_size_candidates) - 1):
                bigger = self.kernel_size_candidates[i]
                smaller = self.kernel_size_candidates[i + 1]
                assert bigger[1] > smaller[1] or (bigger[1] == smaller[1] and bigger[0] > smaller[0]), f'Kernel_size candidates ' \
                    f'should be larger or smaller than each other on both dimensions, but found {bigger} and {smaller}.'
            args['kernel_size'] = self.kernel_size_candidates[0]

        super().__init__(**args)
        self.generate_architecture_params()

    def forward(self, input):
        # Note that there is no need to handle ``in_channels`` here since it is already handle by the ``out_channels`` in the
        # previous module. If we multiply alpha with refer to ``in_channels`` here again, the alpha will indeed be considered
        # twice, which is not what we expect.
        weight = self.weight

        def sum_weight(input_weight, masks, thresholds, indicator):
            """
            This is to get the weighted sum of weight.

            Parameters
            ----------
            input_weight : Tensor
                the weight to be weighted summed
            masks : list[Tensor]
                weight masks.
            thresholds : list[float]
                thresholds, should have a length of ``len(masks) - 1``
            indicator : Callable[[Tensor, float], float]
                take a tensor and a threshold as input, and output the weight

            Returns
            ----------
            weight : Tensor
                weighted sum of ``input_weight``. this is of the same shape as ``input_sum``
            """
            # Note that ``masks`` and ``thresholds`` have different lengths. There alignment is shown below:
            # self.xxx_candidates = [   c_0  ,   c_1  , ... ,  c_n-2  ,   c_n-1 ] # descending order
            # self.xxx_mask       = [ mask_0 , mask_1 , ... , mask_n-2, mask_n-1]
            # self.t_xxx          = [   t_0  ,   t_2  , ... ,  t_n-2 ]
            # So we zip the first n-1 items, and multiply masks[-1] in the end.
            weight = torch.zeros_like(input_weight)
            for mask, t in zip(masks[:-1], thresholds):
                cur_part = input_weight * mask
                alpha = indicator(cur_part, t)
                weight = (weight + cur_part) * alpha
            # we do not consider skip-op here for out_channel/expansion candidates, which means at least the smallest channel
            # candidate is included
            weight += input_weight * masks[-1]

            return weight

        if self.kernel_size_candidates is not None:
            weight = sum_weight(weight, self.kernel_masks, self.t_kernel, self.Lasso_sigmoid)

        if self.out_channel_candidates is not None:
            weight = sum_weight(weight, self.channel_masks, self.t_expansion, self.Lasso_sigmoid)

        output = self._conv_forward(input, weight, self.bias)
        return output

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self):
        for name, p in super().named_parameters():
            if name == 'alpha':
                continue
            yield name, p

    def export(self):
        """
        result = {
            'kernel_size': i,
            'out_channels': j
        }
        which means the best candidate for an argument is the i-th one if candidates are sorted in descending order
        """
        result = {}
        eps = 1e-5
        with torch.no_grad():
            if self.kernel_size_candidates is not None:
                weight = torch.zeros_like(self.weight)
                # ascending order
                for i in range(len(self.kernel_size_candidates) - 2, -1, -1):
                    mask = self.kernel_masks[i]
                    t = self.t_kernel[i]
                    cur_part = self.weight * mask
                    alpha = self.Lasso_sigmoid(cur_part, t)
                    if alpha <= eps:  # takes the smaller one
                        result['kernel_size'] = self.kernel_size_candidates[i + 1]
                        break
                    weight = (weight + cur_part) * alpha

                if 'kernel_size' not in result:
                    result['kernel_size'] = self.kernel_size_candidates[0]
            else:
                weight = self.weight

            if self.out_channel_candidates is not None:
                for i in range(len(self.out_channel_candidates) - 2, -1, -1):
                    mask = self.channel_masks[i]
                    t = self.t_expansion[i]
                    alpha = self.Lasso_sigmoid(weight * mask, t)
                    if alpha <= eps:
                        result['out_channels'] = self.out_channel_candidates[i + 1]

                if 'out_channels' not in result:
                    result['out_channels'] = self.out_channel_candidates[0]

        return result

    @staticmethod
    def Lasso_sigmoid(matrix, t):
        """
        A trick that can make use of both the value of bool(lasso > t) and the gradient of sigmoid(lasso - t)

        Parameters
        ----------
        matrix : Tensor
            the matrix to calculate lasso norm
        t : float
            the threshold
        """
        lasso = torch.norm(matrix) - t
        indicator = (lasso > 0).float()  # torch.sign(lasso)
        with torch.no_grad():
            #            indicator = indicator / 2 + .5 # realign indicator from (-1, 1) to (0, 1)
            indicator -= F.sigmoid(lasso)
        indicator += F.sigmoid(lasso)
        return indicator

    def generate_architecture_params(self):
        self.alpha = {}
        if self.kernel_size_candidates is not None:
            # kernel size arch params
            self.t_kernel = nn.Parameter(torch.rand(len(self.kernel_size_candidates) - 1))
            self.alpha['kernel_size'] = self.t_kernel
            # kernel size mask
            self.kernel_masks = []
            for i in range(0, len(self.kernel_size_candidates) - 1):
                big_size = self.kernel_size_candidates[i]
                small_size = self.kernel_size_candidates[i + 1]
                mask = torch.zeros_like(self.weight)
                mask[:, :, :big_size[0], :big_size[1]] = 1          # if self.weight.shape = (out, in, 7, 7), big_size = (5, 5) and
                mask[:, :, :small_size[0], :small_size[1]] = 0      # small_size = (3, 3), mask will look like:
                self.kernel_masks.append(mask)  # 0 0 0 0 0 0 0
            mask = torch.zeros_like(self.weight)  # 0 1 1 1 1 1 0
            mask[:, :, :self.kernel_size_candidates[-1][0], :self.kernel_size_candidates[-1][1]] = 1  # 0 1 0 0 0 1 0
            self.kernel_masks.append(mask)  # 0 1 0 0 0 1 0
            #   0 1 0 0 0 1 0
        if self.out_channel_candidates is not None:  # 0 1 1 1 1 1 0
            # out_channel (or expansion) arch params. we do not consider skip-op here, so we            #   0 0 0 0 0 0 0
            # only generate ``len(self.kernel_size_candidates) - 1 `` thresholds
            self.t_expansion = nn.Parameter(torch.rand(len(self.out_channel_candidates) - 1))
            self.alpha['out_channels'] = self.t_expansion
            self.channel_masks = []
            for i in range(0, len(self.out_channel_candidates) - 1):
                big_channel, small_channel = self.out_channel_candidates[i], self.out_channel_candidates[i + 1]
                mask = torch.zeros_like(self.weight)
                mask[:big_channel] = 1
                mask[:small_channel] = 0
                # if self.weight.shape = (32, in, W, H), big_channel = 16 and small_size = 8, mask will look like:
                # 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                self.channel_masks.append(mask)
            mask = torch.zeros_like(self.weight)
            mask[:self.out_channel_candidates[-1]] = 1
            self.channel_masks.append(mask)


class DifferentiableBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, module, name):
        self.label = name
        args = module.trace_kwargs
        if isinstance(args['num_features'], ValueChoice):
            args['num_features'] = max(args['num_features'].candidates)
        super().__init__(**args)

        # no architecture parameter is needed for BatchNorm2d Layers
        self.alpha = nn.Parameter(torch.tensor([]))

    def export(self):
        """
        No need to export ``BatchNorm2d``. Refer to the ``Conv2d`` layer that has the ``ValueChoice`` as ``out_channels``.
        """
        return -1
