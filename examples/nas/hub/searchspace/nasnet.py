from .layers import OPS, DropPath_, FactorizedReduce, ReLUConvBN
from configs import NdsConfig, NdsModelType
from common.searchspace import SearchSpace, MixedOp, MixedInput, HyperParameter
from common.dynamic_ops import DynamicBatchNorm2d, DynamicConv2d, DynamicLinear, ResizableSequential
import logging
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper


# the following are NAS operations from
# https://github.com/facebookresearch/unnas/blob/main/pycls/models/nas/operations.py

OPS = {
    'none': lambda C, stride, affine:
        Zero(stride),
    'avg_pool_2x2': lambda C, stride, affine:
        nn.AvgPool2d(2, stride=stride, padding=0, count_include_pad=False),
    'avg_pool_3x3': lambda C, stride, affine:
        nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'avg_pool_5x5': lambda C, stride, affine:
        nn.AvgPool2d(5, stride=stride, padding=2, count_include_pad=False),
    'max_pool_2x2': lambda C, stride, affine:
        nn.MaxPool2d(2, stride=stride, padding=0),
    'max_pool_3x3': lambda C, stride, affine:
        nn.MaxPool2d(3, stride=stride, padding=1),
    'max_pool_5x5': lambda C, stride, affine:
        nn.MaxPool2d(5, stride=stride, padding=2),
    'max_pool_7x7': lambda C, stride, affine:
        nn.MaxPool2d(7, stride=stride, padding=3),
    'skip_connect': lambda C, stride, affine:
        nn.Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'conv_1x1': lambda C, stride, affine:
        nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C, C, 1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(C, affine=affine)
        ),
    'conv_3x3': lambda C, stride, affine:
        nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C, C, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(C, affine=affine)
        ),
    'sep_conv_3x3': lambda C, stride, affine:
        SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine:
        SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine:
        SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine:
        DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine:
        DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'dil_sep_conv_3x3': lambda C, stride, affine:
        DilSepConv(C, C, 3, stride, 2, 2, affine=affine),
    'conv_3x1_1x3': lambda C, stride, affine:
        nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C, C, (1, 3), stride=(1, stride), padding=(0, 1), bias=False),
            nn.Conv2d(C, C, (3, 1), stride=(stride, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(C, affine=affine)
        ),
    'conv_7x1_1x7': lambda C, stride, affine:
        nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
            nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
            nn.BatchNorm2d(C, affine=affine)
        ),
}


class ReLUConvBN(nn.Sequential):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in, C_out, kernel_size, stride=stride,
                padding=padding, bias=False
            ),
            nn.BatchNorm2d(C_out, affine=affine)
        )


class DilConv(nn.Sequential):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in, C_in, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, groups=C_in, bias=False
            ),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )


class SepConv(nn.Sequential):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in, C_in, kernel_size=kernel_size, stride=stride,
                padding=padding, groups=C_in, bias=False
            ),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in, C_in, kernel_size=kernel_size, stride=1,
                padding=padding, groups=C_in, bias=False
            ),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )


class DilSepConv(nn.Sequential):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in, C_in, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, groups=C_in, bias=False
            ),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in, C_in, kernel_size=kernel_size, stride=1,
                padding=padding, dilation=dilation, groups=C_in, bias=False
            ),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )


class Zero(nn.Module):

    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
        self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)

    def forward(self, x):
        x = self.relu(x)
        y = self.pad(x)
        out = torch.cat([self.conv_1(x), self.conv_2(y[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class DropPath_(nn.Module):
    # https://github.com/khanrc/pt.darts/blob/0.1/models/ops.py
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.training and self.drop_prob > 0.:
            keep_prob = 1. - self.drop_prob
            mask = torch.zeros((x.size(0), 1, 1, 1), dtype=torch.float, device=x.device).bernoulli_(keep_prob)
            return x.div(keep_prob).mul(mask)
        return x


class AuxiliaryHead(nn.Module):
    def __init__(self, C: int, num_labels: int, dataset: Literal['imagenet', 'cifar']):
        super().__init__()
        if dataset == 'imagenet':
            # assuming input size 14x14
            stride = 2
        elif dataset == 'cifar':
            stride = 3

        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=stride, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class NdsCell(nn.Module):
    """
    This cell is `nni.retiarii.nn.pytorch.Cell` + shape alignment.

    """

    def __init__(self, n_nodes, primitives, C_prev_prev, C_prev, C, reduction, concat_all):
        super(Cell, self).__init__()
        self.n_nodes = n_nodes
        self.reduction = reduction
        self.cell_type = 'reduce' if reduction else 'normal'
        self.concat_all = concat_all
        self.primitives = primitives

        self.preprocess0_reduce = FactorizedReduce(C_prev_prev, C)
        self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 0)

        self.nodes = nn.ModuleDict()
        self.inputs = nn.ModuleDict()
        for i in range(self.n_nodes):
            self.nodes[f'{i}_x'] = self._build_layer_choice(f'{self.cell_type}_{i}_x', C)
            self.nodes[f'{i}_y'] = self._build_layer_choice(f'{self.cell_type}_{i}_y', C)
            self.inputs[f'{i}_x'] = self._build_input_choice(f'{self.cell_type}_{i}_x', i + 2)
            self.inputs[f'{i}_y'] = self._build_input_choice(f'{self.cell_type}_{i}_y', i + 2)

    def _build_input_choice(self, key, num_input_candidates):
        return MixedInput(key + '_input', num_input_candidates)

    def _build_layer_choice(self, key, channels):
        mapping = collections.OrderedDict()
        for name in self.primitives:
            mapping[name] = OPS[name](channels)
        return MixedOp(key + '_op', mapping)

    def forward(self, s0, s1, width):
        if s0.size(2) != s1.size(2):
            # needs to be down-sampled
            s0 = self.preprocess0_reduce(s0, width)
        else:
            s0 = self.preprocess0(s0, width)
        s1 = self.preprocess1(s1, width)
        states = [s0, s1]
        used_indices = set()
        for i in range(self.n_nodes):
            x_k, y_k = f'{i}_x', f'{i}_y'
            x_reduction = self.reduction and self.inputs[x_k].activated < 2
            y_reduction = self.reduction and self.inputs[y_k].activated < 2
            used_indices |= {self.inputs[x_k].activated, self.inputs[y_k].activated}
            t1 = self.nodes[x_k](self.inputs[x_k](states), width, 2 if x_reduction else 1)
            t2 = self.nodes[y_k](self.inputs[y_k](states), width, 2 if y_reduction else 1)
            states.append(t1 + t2)
        if self.concat_all:
            return torch.cat(states[2:], 1)
        else:
            unused_indices = [i for i in range(2, self.n_nodes + 2) if i not in used_indices]
            return torch.cat([states[i] for i in unused_indices], 1)


def get_cell_builder(op_candidates: List[str], channels: int, num_nodes: int):
    # the cell builder is used in Repeat
    # it takes an index that is the index in the repeat
    def cell_builder(repeat_idx: int):
        # number of predecessors for each cell is fixed to 2.
        num_predecessors = 2
        # number of ops per node is fixed to 2.
        num_ops_per_node = 2

        # reduction cell means stride = 2.
        if repeat_idx == 0:
            cell_type = 'reduction'
        else:
            cell_type = 'normal'

        ops_factory = [
            lambda node_index, op_index, input_index: \
                OPS[op](channels, 2 if cell_type == 'reduction' and input_index < num_predecessors else 1, True)
            for op in op_candidates
        ]

        return nn.Cell(ops_factory, num_nodes, num_ops_per_node, num_predecessors, 'loose_end',
                       preprocesser=, label=cell_type)

    return cell_builder


@model_wrapper
class NDS(nn.Module):
    """
    The unified version of NASNet search space, implemented in
    `unnas <https://github.com/facebookresearch/unnas/blob/main/pycls/models/nas/nas.py>`__.
    See [nds] for details.

    Different NAS papers usually differ in the way that they specify ``op_candidates`` and ``merge_op``.
    ``dataset`` here is to give a hint about input resolution, so as to create reasonable stem and auxiliary heads.

    [nds] has a speciality that it has mutable depths/widths.
    This is implemented by accepting a list of int as ``num_cells`` / ``width``.

    .. [nds] Radosavovic, Ilija and Johnson, Justin and Xie, Saining and Lo, Wan-Yen and Dollar, Piotr,
         "On Network Design Spaces for Visual Recognition". https://arxiv.org/abs/1905.13214
    """

    def __init__(self,
                 op_candidates: List[str],
                 merge_op: Literal['all', 'loose_end'] = 'all',
                 num_labels: int = 10,
                 width: Union[List[int], int] = 16,
                 num_cells: Union[List[int], int] = 20,
                 dataset: Literal['cifar', 'imagenet'] = 'imagenet'):

        # preprocess the specified width and depth
        if isinstance(width, list):
            C = nn.ValueChoice(width, label='width')
        else:
            C = width

        if isinstance(num_cells, list):
            num_cells = nn.ValueChoice(num_cells, label='depth')
        num_cells_per_stage = [i * num_cells // 3 - (i - 1) * num_cells // 3 for i in range(3)]


        # auxiliary head is different for network targetted at different datasets
        if dataset == 'imagenet':
            self.stem0 = nn.Sequential(
                nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C),
            )
            self.stem1 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C),
            )
            C_prev_prev, C_prev, C_curr = C, C, C
        elif dataset == 'cifar':
            self.stem = nn.Sequential(
                nn.Conv2d(3, 3 * C, 3, padding=1, bias=False),
                nn.BatchNorm2d(3 * C)
            )
            C_prev_prev, C_prev, C_curr = 3 * C, 3 * C, C

        self.stages = nn.ModuleList()
        for stage_idx in range(3):
            cell_builder = get_cell_builder(op_candidates, C_curr, )
            self.stages.append(nn.Repeat(, num_cells_per_stage[stage_idx]))
            if stage_idx > 0:
                C_curr *= 2
            stage = nn.ModuleList()
            for i in range((self.max_num_cells + 2) // 3):  # div and ceil
                cell = nn.Cell(op_candidates, config.n_nodes, op_candidates, C_prev_prev, C_prev, C_curr,
                            stage_idx > 0 and i == 0, )
                stage.append(cell)
                C_prev_prev, C_prev = C_prev, cell.n_nodes * C_curr
            if stage_idx == 2:
                C_to_auxiliary = C_prev
            self.stages.append(stage)

        if self.use_aux:
            self.auxiliary_head = AuxiliaryHead(C_to_auxiliary, self.num_labels)
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = DynamicLinear(C_prev, self.num_labels)

    def forward(self, inputs):
        width = self.width_selector()
        depth = self.depth_selector()
        if self.model_type == NdsModelType.ImageNet:
            s0 = self.stem0(inputs, width / self.max_init_channels)
            s1 = self.stem1(s0, width / self.max_init_channels)
        else:
            s0 = s1 = self.stem(inputs, width / self.max_init_channels)

        cur_stage, cur_idx = 0, 0
        for i in range(depth):
            if i in [depth // 3, 2 * depth // 3]:
                width *= 2
                cur_stage += 1
                cur_idx = 0
            s0, s1 = s1, self.stages[cur_stage][cur_idx](s0, s1, width)
            if i == 2 * depth // 3:
                if self.training and self.use_aux:
                    logits_aux = self.auxiliary_head(s1)
            cur_idx += 1
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        if self.training and self.use_aux:
            return logits, logits_aux
        else:
            return logits

    def set_drop_path_prob(self, drop_prob):
        for module in self.modules():
            if isinstance(module, DropPath_):
                module.drop_prob = drop_prob
