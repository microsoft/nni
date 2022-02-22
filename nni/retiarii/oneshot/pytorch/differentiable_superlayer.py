from cgitb import small
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
        self.args = module.trace_kwargs

        # compulsory params
        max_in_channel = self.args['in_channels']
        if isinstance(max_in_channel, ValueChoice):
            max_in_channel = max(max_in_channel.candidates)

        max_out_channel  = self.args['out_channels']
        self.out_channel_candidates = None
        if isinstance(max_out_channel, ValueChoice):
            self.out_channel_candidates = sorted(max_out_channel.candidates, reverse=True)
            max_out_channel =  self.out_channel_candidates[0]

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
            indicator = torch.sign(lasso)
            with torch.no_grad():
                indicator = indicator / 2 + .5 # realign indicator from (-1, 1) to (0, 1)
                indicator -= F.sigmoid(lasso)
            indicator += F.sigmoid(lasso)
            return indicator

        def weighted_sum_weight(input_weight, masks, thresholds, indicator):
            """
            This is to get the weighted sum of weight.

            Parameters
            ----------
            input_weight : Tensor
                the weight to be weighted summed
            masks : List[Tensor]
                weight masks.
            thresholds : List[float]
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
            weight += input_weight * mask[-1]

            return weight

        if self.kernel_size_candidates is not None:
            weight = weighted_sum_weight(weight, self.kernel_masks, self.t_kernel, Lasso_sigmoid)

        if self.out_channel_candidates is not None:
            weight = weighted_sum_weight(weight, self.channel_masks, self.t_expansion, Lasso_sigmoid)

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
        result = {}
        for k, v in self.alpha.items():
            result[k] = torch.argsort(v).cpu().numpy().tolist()[0]
        return result

    def pre_process_kernel_size(self):
        """
        Unify kernel size candidates as tuples and sort them in descending order if kernel size is a ``ValueChoice``.
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

        # sort kernel size in descending order
        self.kernel_size_candidates = sorted(candidates, key=lambda t : t[0], reverse=True)
        for i in range(0, len(self.kernel_size_candidates) - 1):
            bigger = self.kernel_size_candidates[i]
            smaller = self.kernel_size_candidates[i + 1]
            assert bigger[1] > smaller[1] or (bigger[1] == smaller[1] and bigger[0] > smaller[0]), f'Kernel_size candidates ' \
                f'should be larger or smaller than each other on both dimensions, but found {bigger} and {smaller}.'
        return self.kernel_size_candidates[0]


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
                self.kernel_masks.append(mask)                                                          #   0 0 0 0 0 0 0
            mask = torch.zeros_like(self.weight)                                                        #   0 1 1 1 1 1 0
            mask[:, :, :self.kernel_size_candidates[-1][0], :self.kernel_size_candidates[-1][1]] = 1     #   0 1 0 0 0 1 0
            self.kernel_masks.append(mask)                                                              #   0 1 0 0 0 1 0
                                                                                                        #   0 1 0 0 0 1 0
        if self.out_channel_candidates is not None:                                                     #   0 1 1 1 1 1 0
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
