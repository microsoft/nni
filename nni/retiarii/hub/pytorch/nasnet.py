# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""File containing NASNet-series search space.

The implementation is based on NDS.
It's called ``nasnet.py`` simply because NASNet is the first to propose such structure.
"""

from collections import OrderedDict
from typing import Tuple, List, Union, Iterable, Dict, Callable

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
        if isinstance(C_out, int):
            assert C_out % 2 == 0
        else:   # is a value choice
            assert all(c % 2 == 0 for c in C_out.all_options())
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


class SequentialBreakdown(nn.Sequential):
    """Return all layers of a sequential."""

    def __init__(self, sequential: nn.Sequential):
        super().__init__(OrderedDict(sequential.named_children()))

    def forward(self, inputs):
        result = []
        for module in self:
            inputs = module(inputs)
            result.append(inputs)
        return result


class CellPreprocessor(nn.Module):
    """
    Aligning the shape of predecessors.

    If the last cell is a reduction cell, ``pre0`` should be ``FactorizedReduce`` instead of ``ReLUConvBN``.
    See :class:`CellBuilder` on how to calculate those channel numbers.
    """

    def __init__(self, C_pprev: int, C_prev: int, C: int, last_cell_reduce: bool) -> None:
        super().__init__()

        if last_cell_reduce:
            self.pre0 = FactorizedReduce(C_pprev, C)
        else:
            self.pre0 = ReLUConvBN(C_pprev, C, 1, 1, 0)
        self.pre1 = ReLUConvBN(C_prev, C, 1, 1, 0)

    def forward(self, cells):
        assert len(cells) == 2
        pprev, prev = cells
        pprev = self.pre0(pprev)
        prev = self.pre1(prev)

        return [pprev, prev]


class CellPostprocessor(nn.Module):
    """
    The cell outputs previous cell + this cell, so that cells can be directly chained.
    """

    def forward(self, this_cell, previous_cells):
        return [previous_cells[-1], this_cell]


class CellBuilder:
    """The cell builder is used in Repeat.
    Builds an cell each time it's "called".
    Note that the builder is ephemeral, it can only be called once for every index.
    """

    def __init__(self, op_candidates: List[str], C_prev_in: int, C_in: int, C: int,
                 num_nodes: int, merge_op: Literal['all', 'loose_end'],
                 first_cell_reduce: bool, last_cell_reduce: bool):
        self.C_prev_in = C_prev_in      # This is the out channels of the cell before last cell.
        self.C_in = C_in                # This is the out channesl of last cell.
        self.C = C                      # This is NOT C_out of this stage, instead, C_out = C * len(cell.output_node_indices)
        self.op_candidates = op_candidates
        self.num_nodes = num_nodes
        self.merge_op = merge_op
        self.first_cell_reduce = first_cell_reduce
        self.last_cell_reduce = last_cell_reduce
        self._expect_idx = 0

    def __call__(self, repeat_idx: int):
        if self._expect_idx != repeat_idx:
            raise ValueError(f'Expect index {self._expect_idx}, found {repeat_idx}')

        # It takes an index that is the index in the repeat.
        # Number of predecessors for each cell is fixed to 2.
        num_predecessors = 2
        # Number of ops per node is fixed to 2.
        num_ops_per_node = 2

        # Reduction cell means stride = 2 and channel multiplied by 2.
        is_reduction_cell = repeat_idx == 0 and self.first_cell_reduce

        # self.C_prev_in, self.C_in, self.last_cell_reduce are updated after each cell is built.
        preprocessor = CellPreprocessor(self.C_prev_in, self.C_in, self.C, self.last_cell_reduce)

        ops_factory: Dict[str, Callable[[int, int, int], nn.Module]] = {
            op:  # make final chosen ops named with their aliases
            lambda node_index, op_index, input_index:
            OPS[op](self.C, 2 if is_reduction_cell and (
                    input_index is None or input_index < num_predecessors  # could be none when constructing search sapce
                    ) else 1, True)
            for op in self.op_candidates
        }

        cell = nn.Cell(ops_factory, self.num_nodes, num_ops_per_node, num_predecessors, self.merge_op,
                       preprocessor=preprocessor, postprocessor=CellPostprocessor(),
                       label='reduce' if is_reduction_cell else 'normal')

        # update state
        self.C_prev_in = self.C_in
        self.C_in = self.C * len(cell.output_node_indices)
        self.last_cell_reduce = is_reduction_cell
        self._expect_idx += 1

        return cell


_INIT_PARAMETER_DOCS = """

    Parameters
    ----------
    width : int or tuple of int
        A fixed initial width or a tuple of widths to choose from.
    num_cells : int or tuple of int
        A fixed number of cells (depths) to stack, or a tuple of depths to choose from.
    dataset : "cifar" | "imagenet"
        The essential differences are in "stem" cells, i.e., how they process the raw image input.
        Choosing "imagenet" means more downsampling at the beginning of the network.
    auxiliary_loss : bool
        If true, another auxiliary classification head will produce the another prediction.
        This makes the output of network two logits in the training phase.

"""


class NDS(nn.Module):
    """
    The unified version of NASNet search space.

    We follow the implementation in
    `unnas <https://github.com/facebookresearch/unnas/blob/main/pycls/models/nas/nas.py>`__.
    See `On Network Design Spaces for Visual Recognition <https://arxiv.org/abs/1905.13214>`__ for details.

    Different NAS papers usually differ in the way that they specify ``op_candidates`` and ``merge_op``.
    ``dataset`` here is to give a hint about input resolution, so as to create reasonable stem and auxiliary heads.

    NDS has a speciality that it has mutable depths/widths.
    This is implemented by accepting a list of int as ``num_cells`` / ``width``.
    """ + _INIT_PARAMETER_DOCS + """
    op_candidates : list of str
        List of operator candidates. Must be from ``OPS``.
    merge_op : ``all`` or ``loose_end``
        See :class:`~nni.retiarii.nn.pytorch.Cell`.
    num_nodes_per_cell : int
        See :class:`~nni.retiarii.nn.pytorch.Cell`.
    """

    def __init__(self,
                 op_candidates: List[str],
                 merge_op: Literal['all', 'loose_end'] = 'all',
                 num_nodes_per_cell: int = 4,
                 width: Union[Tuple[int], int] = 16,
                 num_cells: Union[Tuple[int], int] = 20,
                 dataset: Literal['cifar', 'imagenet'] = 'imagenet',
                 auxiliary_loss: bool = False):
        super().__init__()

        self.dataset = dataset
        self.num_labels = 10 if dataset == 'cifar' else 1000
        self.auxiliary_loss = auxiliary_loss

        # preprocess the specified width and depth
        if isinstance(width, Iterable):
            C = nn.ValueChoice(list(width), label='width')
        else:
            C = width

        if isinstance(num_cells, Iterable):
            num_cells = nn.ValueChoice(list(num_cells), label='depth')
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
            C_pprev = C_prev = C_curr = C
            last_cell_reduce = True
        elif dataset == 'cifar':
            self.stem = nn.Sequential(
                nn.Conv2d(3, 3 * C, 3, padding=1, bias=False),
                nn.BatchNorm2d(3 * C)
            )
            C_pprev = C_prev = 3 * C
            C_curr = C
            last_cell_reduce = False

        self.stages = nn.ModuleList()
        for stage_idx in range(3):
            if stage_idx > 0:
                C_curr *= 2
            # For a stage, we get C_in, C_curr, and C_out.
            # C_in is only used in the first cell.
            # C_curr is number of channels for each operator in current stage.
            # C_out is usually `C * num_nodes_per_cell` because of concat operator.
            cell_builder = CellBuilder(op_candidates, C_pprev, C_prev, C_curr, num_nodes_per_cell,
                                       merge_op, stage_idx > 0, last_cell_reduce)
            stage = nn.Repeat(cell_builder, num_cells_per_stage[stage_idx])
            self.stages.append(stage)

            # C_pprev is output channel number of last second cell among all the cells already built.
            if len(stage) > 1:
                # Contains more than one cell
                C_pprev = len(stage[-2].output_node_indices) * C_curr
            else:
                # Look up in the out channels of last stage.
                C_pprev = C_prev

            # This was originally,
            # C_prev = num_nodes_per_cell * C_curr.
            # but due to loose end, it becomes,
            C_prev = len(stage[-1].output_node_indices) * C_curr

            # Useful in aligning the pprev and prev cell.
            last_cell_reduce = cell_builder.last_cell_reduce

            if stage_idx == 2:
                C_to_auxiliary = C_prev

        if auxiliary_loss:
            assert isinstance(self.stages[2], nn.Sequential), 'Auxiliary loss can only be enabled in retrain mode.'
            self.stages[2] = SequentialBreakdown(self.stages[2])
            self.auxiliary_head = AuxiliaryHead(C_to_auxiliary, self.num_labels, dataset=self.dataset)

        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(C_prev, self.num_labels)

    def forward(self, inputs):
        if self.dataset == 'imagenet':
            s0 = self.stem0(inputs)
            s1 = self.stem1(s0)
        else:
            s0 = s1 = self.stem(inputs)

        for stage_idx, stage in enumerate(self.stages):
            if stage_idx == 2 and self.auxiliary_loss:
                s = list(stage([s0, s1]).values())
                s0, s1 = s[-1]
                if self.training:
                    # auxiliary loss is attached to the first cell of the last stage.
                    logits_aux = self.auxiliary_head(s[0][1])
            else:
                s0, s1 = stage([s0, s1])

        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        if self.training and self.auxiliary_loss:
            return logits, logits_aux
        else:
            return logits

    def set_drop_path_prob(self, drop_prob):
        """
        Set the drop probability of Drop-path in the network.
        Reference: `FractalNet: Ultra-Deep Neural Networks without Residuals <https://arxiv.org/pdf/1605.07648v4.pdf>`__.
        """
        for module in self.modules():
            if isinstance(module, DropPath_):
                module.drop_prob = drop_prob


@model_wrapper
class NASNet(NDS):
    __doc__ = """
    Search space proposed in `Learning Transferable Architectures for Scalable Image Recognition <https://arxiv.org/abs/1707.07012>`__.

    It is built upon :class:`~nni.retiarii.nn.pytorch.Cell`, and implemented based on :class:`~NDS`.
    Its operator candidates are :attribute:`~NASNet.NASNET_OPS`.
    It has 5 nodes per cell, and the output is concatenation of nodes not used as input to other nodes.
    """ + _INIT_PARAMETER_DOCS

    NASNET_OPS = [
        'skip_connect',
        'conv_3x1_1x3',
        'conv_7x1_1x7',
        'dil_conv_3x3',
        'avg_pool_3x3',
        'max_pool_3x3',
        'max_pool_5x5',
        'max_pool_7x7',
        'conv_1x1',
        'conv_3x3',
        'sep_conv_3x3',
        'sep_conv_5x5',
        'sep_conv_7x7',
    ]

    def __init__(self,
                 width: Union[Tuple[int], int] = (16, 24, 32),
                 num_cells: Union[Tuple[int], int] = (4, 8, 12, 16, 20),
                 dataset: Literal['cifar', 'imagenet'] = 'cifar',
                 auxiliary_loss: bool = False):
        super().__init__(self.NASNET_OPS,
                         merge_op='loose_end',
                         num_nodes_per_cell=5,
                         width=width,
                         num_cells=num_cells,
                         dataset=dataset,
                         auxiliary_loss=auxiliary_loss)


@model_wrapper
class ENAS(NDS):
    __doc__ = """Search space proposed in `Efficient neural architecture search via parameter sharing <https://arxiv.org/abs/1802.03268>`__.

    It is built upon :class:`~nni.retiarii.nn.pytorch.Cell`, and implemented based on :class:`~NDS`.
    Its operator candidates are :attribute:`~ENAS.ENAS_OPS`.
    It has 5 nodes per cell, and the output is concatenation of nodes not used as input to other nodes.
    """ + _INIT_PARAMETER_DOCS

    ENAS_OPS = [
        'skip_connect',
        'sep_conv_3x3',
        'sep_conv_5x5',
        'avg_pool_3x3',
        'max_pool_3x3',
    ]

    def __init__(self,
                 width: Union[Tuple[int], int] = (16, 24, 32),
                 num_cells: Union[Tuple[int], int] = (4, 8, 12, 16, 20),
                 dataset: Literal['cifar', 'imagenet'] = 'cifar',
                 auxiliary_loss: bool = False):
        super().__init__(self.ENAS_OPS,
                         merge_op='loose_end',
                         num_nodes_per_cell=5,
                         width=width,
                         num_cells=num_cells,
                         dataset=dataset,
                         auxiliary_loss=auxiliary_loss)


@model_wrapper
class AmoebaNet(NDS):
    __doc__ = """Search space proposed in
    `Regularized evolution for image classifier architecture search <https://arxiv.org/abs/1802.01548>`__.

    It is built upon :class:`~nni.retiarii.nn.pytorch.Cell`, and implemented based on :class:`~NDS`.
    Its operator candidates are :attribute:`~AmoebaNet.AMOEBA_OPS`.
    It has 5 nodes per cell, and the output is concatenation of nodes not used as input to other nodes.
    """ + _INIT_PARAMETER_DOCS

    AMOEBA_OPS = [
        'skip_connect',
        'sep_conv_3x3',
        'sep_conv_5x5',
        'sep_conv_7x7',
        'avg_pool_3x3',
        'max_pool_3x3',
        'dil_sep_conv_3x3',
        'conv_7x1_1x7',
    ]

    def __init__(self,
                 width: Union[Tuple[int], int] = (16, 24, 32),
                 num_cells: Union[Tuple[int], int] = (4, 8, 12, 16, 20),
                 dataset: Literal['cifar', 'imagenet'] = 'cifar',
                 auxiliary_loss: bool = False):

        super().__init__(self.AMOEBA_OPS,
                         merge_op='loose_end',
                         num_nodes_per_cell=5,
                         width=width,
                         num_cells=num_cells,
                         dataset=dataset,
                         auxiliary_loss=auxiliary_loss)


@model_wrapper
class PNAS(NDS):
    __doc__ = """Search space proposed in
    `Progressive neural architecture search <https://arxiv.org/abs/1712.00559>`__.

    It is built upon :class:`~nni.retiarii.nn.pytorch.Cell`, and implemented based on :class:`~NDS`.
    Its operator candidates are :attribute:`~PNAS.PNAS_OPS`.
    It has 5 nodes per cell, and the output is concatenation of all nodes in the cell.
    """ + _INIT_PARAMETER_DOCS

    PNAS_OPS = [
        'sep_conv_3x3',
        'sep_conv_5x5',
        'sep_conv_7x7',
        'conv_7x1_1x7',
        'skip_connect',
        'avg_pool_3x3',
        'max_pool_3x3',
        'dil_conv_3x3',
    ]

    def __init__(self,
                 width: Union[Tuple[int], int] = (16, 24, 32),
                 num_cells: Union[Tuple[int], int] = (4, 8, 12, 16, 20),
                 dataset: Literal['cifar', 'imagenet'] = 'cifar',
                 auxiliary_loss: bool = False):
        super().__init__(self.PNAS_OPS,
                         merge_op='all',
                         num_nodes_per_cell=5,
                         width=width,
                         num_cells=num_cells,
                         dataset=dataset,
                         auxiliary_loss=auxiliary_loss)


@model_wrapper
class DARTS(NDS):
    __doc__ = """Search space proposed in `Darts: Differentiable architecture search <https://arxiv.org/abs/1806.09055>`__.

    It is built upon :class:`~nni.retiarii.nn.pytorch.Cell`, and implemented based on :class:`~NDS`.
    Its operator candidates are :attribute:`~DARTS.DARTS_OPS`.
    It has 4 nodes per cell, and the output is concatenation of all nodes in the cell.
    """ + _INIT_PARAMETER_DOCS

    DARTS_OPS = [
        'none',
        'max_pool_3x3',
        'avg_pool_3x3',
        'skip_connect',
        'sep_conv_3x3',
        'sep_conv_5x5',
        'dil_conv_3x3',
        'dil_conv_5x5',
    ]

    def __init__(self,
                 width: Union[Tuple[int], int] = (16, 24, 32),
                 num_cells: Union[Tuple[int], int] = (4, 8, 12, 16, 20),
                 dataset: Literal['cifar', 'imagenet'] = 'cifar',
                 auxiliary_loss: bool = False):
        super().__init__(self.DARTS_OPS,
                         merge_op='all',
                         num_nodes_per_cell=4,
                         width=width,
                         num_cells=num_cells,
                         dataset=dataset,
                         auxiliary_loss=auxiliary_loss)
