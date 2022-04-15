# This code is modified from https://github.com/huggingface/transformers/tree/master/examples/research_projects/movement-pruning
# Licensed under the Apache License, Version 2.0 (the "License");
# We add more functionalities as well as remove unnecessary functionalities
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from .binarizer import TopKBinarizer

# This function is from
# https://stackoverflow.com/questions/16873441/form-a-big-2d-array-from-multiple-smaller-2d-arrays


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array looks like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h // nrows, nrows, -1, ncols)
               .swapaxes(1, 2)
               .reshape(-1, nrows, ncols))

# This function is from
# https://stackoverflow.com/questions/16873441/form-a-big-2d-array-from-multiple-smaller-2d-arrays


def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h // nrows, -1, nrows, ncols)
               .swapaxes(1, 2)
               .reshape(h, w))


class MaskedLinear(nn.Linear):
    """
    Fully Connected layer with on the fly adaptive mask during training,
    and does real pruning during inference
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mask_init: str = "constant",
        mask_scale: float = 0.0,
        pruning_method: str = "topK",
        head_split: int = -1,
        bias_mask: bool = False,
        head_pruning: bool = False,
        row_pruning: bool = True
    ):
        """
        Args:
            in_features (`int`)
                Size of each input sample
            out_features (`int`)
                Size of each output sample
            bias (`bool`)
                If set to ``False``, the layer will not learn an additive bias.
                Default: ``True``
            mask_init (`str`)
                The initialization method for the score matrix if a score matrix is needed.
                Choices: ["constant", "uniform", "kaiming"]
                Default: ``constant``
            mask_scale (`float`)
                The initialization parameter for the chosen initialization method `mask_init`.
                Default: ``0.``
            pruning_method (`str`)
                Method to compute the mask.
                Default: ``topK``
            head_split:
                The number of head in the layer. This can also used to make each head prune
                out with same number of rows (so that we can do parallize forward with reshape)
                Default: ``-1`` (means no need for head split)
            bias_mask:
                Prune bias or not
                Default: False
            head_pruning:
                Do Head Pruning or not
                Default: False
            row_pruning:
                Do Row Pruning or Not
                Defualt: True
        """
        super(
            MaskedLinear,
            self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias)

        self.pruning_method = pruning_method
        self.head_split = head_split
        self.bias_mask = bias_mask
        self.head_pruning = head_pruning
        self.row_pruning = row_pruning

        self.inference_mode = False
        # this is used for final block-wise pruning, for init we do not need to
        # worry about that!
        self.block_pruning = False  # We will enable this when needed
        self.block_mask_scores = None  # the mask for block wise pruning
        self.threshold_block = None  # the threshold for block wise pruning

        self.mask_scale = mask_scale
        self.mask_init = mask_init
        if self.row_pruning:
            self.mask_scores = nn.Parameter(
                torch.Tensor(
                    self.weight.size(0),
                    1))  # number of rows * 1
            self.init_mask(self.mask_scores)
            self.threshold_row = nn.Parameter(torch.zeros(1) + 10.0)
        if self.head_pruning:
            self.head_mask_scores = nn.Parameter(
                torch.Tensor(self.head_split, 1))  # number of heads * 1
            self.init_mask(self.head_mask_scores)
            self.threshold_head = nn.Parameter(torch.zeros(1) + 10.0)

    def init_mask(self, mask):
        if self.mask_init == "constant":
            init.constant_(mask, val=self.mask_scale)
        elif self.mask_init == "uniform":
            init.uniform_(mask, a=-self.mask_scale, b=self.mask_scale)
        elif self.mask_init == "kaiming":
            init.kaiming_uniform_(mask, a=math.sqrt(5))

    def get_mask(self):
        # get head mask
        if self.head_pruning:
            mask_head = TopKBinarizer.apply(
                self.head_mask_scores, self.threshold_head, -1)  # for now, only support this
        else:
            mask_head = None

        if self.row_pruning:
            mask = TopKBinarizer.apply(
                self.mask_scores, self.threshold_row, -1)
        else:
            mask = None
        return mask_head, mask

    def make_inference_pruning(self, blocksize):
        self.inference_mode = True
        weight_shape = self.weight.size()
        # if there is no block wise pruning needed, we do not have to increase the
        # numner of rows/cols.
        # Otherwise, we need to pad the matrix so that the # of ros/cols is divided by
        # block size
        if blocksize[0] is not None and blocksize[1] is not None and self.row_pruning:
            rows = weight_shape[0]
            row_block = blocksize[0]
            # remain rows
            mask_head, mask = self.get_mask()
            remaining_row_block = math.ceil(mask.sum().item() / row_block)
            remaining_row = remaining_row_block * row_block
            remaining_ratio = remaining_row / rows - 1e-6
            self.threshold_row.data = self.threshold_row.data * 0 + \
                math.log(remaining_ratio / (1 - remaining_ratio))

        mask_head, mask = self.get_mask()
        if not self.head_pruning:
            mask_head = torch.ones_like(self.weight[:, 0]).type(
                'torch.BoolTensor').view(-1)
        else:
            mask_head = mask_head.type('torch.BoolTensor').view(-1)
            mask_head = torch.repeat_interleave(
                mask_head, weight_shape[0] // self.head_split)
        if not self.row_pruning:
            mask = torch.ones_like(self.weight[:, 0])

        mask = mask.type('torch.BoolTensor').view(-1)
        mask = torch.logical_and(mask_head, mask)
        self.weight = nn.Parameter(self.weight[mask, :])
        if self.bias_mask:
            self.bias = nn.Parameter(self.bias[mask])

        # we do not need those parameters!
        self.mask_scores = None
        self.head_mask_scores = None
        self.threshold_head = None
        self.threshold_row = None
        # we need this mask for some Layer O and FC2 pruning
        return mask

    def make_column_purning(self, mask):
        # make column pruning for Layer O and FC2
        self.weight = nn.Parameter(self.weight[:, mask])

    def enable_block_pruning(self, block_size):
        # As the name suggested, enable block wise pruning
        self.block_pruning = True
        self.block_rows = block_size[0]
        self.block_cols = block_size[1]
        mask_size_row = self.weight.size(0) // block_size[0]
        mask_size_col = self.weight.size(1) // block_size[1]
        self.block_mask_scores = nn.Parameter(
            torch.Tensor(
                mask_size_row *
                mask_size_col,
                1,
                1))  # number of row_block * col_block
        self.init_mask(self.block_mask_scores)
        self.threshold_block = nn.Parameter(torch.zeros(1) + 10.0)

    def get_block_wise_pruning(self):
        # As the name suggested, get the block wise mask
        mask_block = TopKBinarizer.apply(
            self.block_mask_scores, self.threshold_block, -1)  # for now, only support this
        return mask_block

    def make_block_wise_inference_pruning(self):
        self.runsparse = True
        real_param_count = True
        if self.runsparse:
            return self._make_block_wise_inference_pruning_sparse(
                real_param_count)
        else:
            return self._make_block_wise_inference_pruning_base()

    def _make_block_wise_inference_pruning_sparse(self, real_param_count):
        import triton

        assert self.block_rows == self.block_cols
        block = self.block_rows
        assert block in [16, 32]
        threshold = 0.4
        if block == 16:
            threshold = 0.2

        mask_size_row = self.weight.size(0) // block
        mask_size_col = self.weight.size(1) // block

        mask_block = self.get_block_wise_pruning()

        if mask_block.sum() / mask_size_row / mask_size_col < threshold:
            mask_block = mask_block.reshape(
                (mask_size_row, mask_size_col)).type(
                torch.LongTensor)
            mask_block = torch.transpose(
                mask_block, 0, 1).reshape(
                (1, mask_size_col, mask_size_row))
            self.op = triton.ops.blocksparse.matmul(
                mask_block, block, "dds", trans_a=False, trans_b=False)
            tmp_weight = torch.transpose(self.weight, 0, 1).reshape(
                (1, 1, self.weight.shape[1], self.weight.shape[0]))
            
            self.sparse_weight = triton.testing.sparsify_tensor(
                tmp_weight, mask_block, block)
            self.sparse_weight = nn.Parameter(self.sparse_weight)
            tmp_ori_weight = self.weight.data
            tmp_ori_weight = blockshaped(tmp_ori_weight, self.block_rows, self.block_cols)
            # import pdb; pdb.set_trace()
            tmp_ori_weight = tmp_ori_weight * self.get_block_wise_pruning()
            self.ori_weight = unblockshaped(tmp_ori_weight, self.weight.size(0), self.weight.size(1)).to(self.weight.device)
            self.sparse_kernel = True
            self.weight = self.sparse_weight
            
            self.run_sparse = False
            self.weight = nn.Parameter(self.ori_weight)
        else:
            self.sparse_kernel = False
            rows, cols = self.weight.shape
            tmp_weight = blockshaped(
                self.weight,
                self.block_rows,
                self.block_cols)  # n-block x 32 x 32
            tmp_weight = tmp_weight * mask_block  # n-block x 1 x 1
            tmp_weight = unblockshaped(tmp_weight, rows, cols)  # d x d
            self.ori_weight = tmp_weight
            if not real_param_count:
                self.weight = nn.Parameter(tmp_weight)

        # we do not need those values anymore
        self.block_pruning = False
        self.block_mask_scores = None
        self.threshold_block = None

    def _make_block_wise_inference_pruning_base(self):
        mask_block = self.get_block_wise_pruning()
        rows, cols = self.weight.shape
        tmp_weight = blockshaped(
            self.weight,
            self.block_rows,
            self.block_cols)  # n-block x 32 x 32
        tmp_weight = tmp_weight * mask_block  # n-block x 1 x 1
        tmp_weight = unblockshaped(tmp_weight, rows, cols)  # d x d
        self.weight = nn.Parameter(tmp_weight)
        # we do not need those values anymore
        self.block_pruning = False
        self.block_mask_scores = None
        self.threshold_block = None

    def forward(self, input: torch.tensor):
        if not self.inference_mode:
            output = self.training_forward(input)
        else:
            if not self.block_pruning:
                output = self.inference_forward(input)
            else:
                output = self.block_pruning_forward(input)
        return output

    def block_pruning_forward(self, input: torch.tensor):
        mask_block = self.get_block_wise_pruning()
        rows, cols = self.weight.shape
        tmp_weight = blockshaped(self.weight, self.block_rows, self.block_cols)
        tmp_weight = tmp_weight * mask_block
        tmp_weight = unblockshaped(tmp_weight, rows, cols)

        return F.linear(input, tmp_weight, self.bias)

    def inference_forward(self, input: torch.tensor):
        if self.runsparse and self.sparse_kernel:
            a, b, c = input.shape[0], input.shape[1], input.shape[2]
            input = input.view((1, 1, a * b, c))
            out = self.op(input, self.sparse_weight)
            out = out.view((a, b, out.shape[-1]))
            return out + self.bias
        else:
            return F.linear(input, self.weight, self.bias)

    def training_forward(self, input: torch.tensor):
        mask_head, mask = self.get_mask()

        weight_shape = self.weight.size()
        bias_shape = self.bias.size()
        if self.head_pruning:
            weight_thresholded = (
                self.weight.view(
                    self.head_split, -1) * mask_head).view(weight_shape)
            if self.bias_mask:
                bias_thresholded = (
                    self.bias.view(
                        self.head_split, -1) * mask_head).view(bias_shape)
        else:
            weight_thresholded = self.weight
            bias_thresholded = self.bias
        # Mask weights with computed mask
        if self.row_pruning:
            weight_thresholded = mask * weight_thresholded
            if self.bias_mask:
                bias_thresholded = mask.view(
                    self.bias.size()) * bias_thresholded
            else:
                bias_thresholded = bias_thresholded

        return F.linear(input, weight_thresholded, bias_thresholded)
