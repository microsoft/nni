from collections import OrderedDict

import torch
import torch.nn as nn
from sdk.graph import Graph
from sdk.mutators import Mutator
from sdk.translate_code import gen_pytorch_graph

from .blocks import *
from ..common import SimpleDifferentiableOpChoice, SimpleDifferentiableTensorChoice


__all__ = ['nasnet', 'amoebanet', 'pnas', 'Cell']


_NASNET_A_NORMAL = {
    'operations': ['sepconv5x5', 'sepconv3x3', 'sepconv5x5', 'sepconv3x3', 'avgpool3x3',
                   'identity', 'avgpool3x3', 'avgpool3x3', 'sepconv3x3', 'identity'],
    'hidden_indices': [0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
    'concat_indices': [1, 2, 3, 4, 5, 6],
}


_NASNET_A_REDUCE = {
    'operations': ['sepconv5x5', 'sepconv7x7', 'maxpool3x3', 'sepconv7x7', 'avgpool3x3',
                   'sepconv5x5', 'identity', 'avgpool3x3', 'sepconv3x3', 'maxpool3x3'],
    'hidden_indices': [0, 1, 0, 1, 0, 1, 3, 2, 2, 0],
    'concat_indices': [3, 4, 5, 6],
}


def _factorize_reduce_build(module, in_ch, out_ch):
    # Ugly. To align with existing pretrained checkpoints.
    module.path_1 = nn.Sequential()
    module.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
    module.path_1.add_module('conv', nn.Conv2d(in_ch, out_ch // 2, 1, stride=1, bias=False))
    module.path_2 = nn.Sequential()
    module.path_2.add_module('padding', nn.ZeroPad2d((0, 1, 0, 1)))  # to avoid nan issue, which I don't know why
    module.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
    module.path_2.add_module('conv', nn.Conv2d(in_ch, (out_ch + 1) // 2, 1, stride=1, bias=False))
    module.final_path_bn = nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.1, affine=True)


def _factorize_reduce_forward(module, x):
    x_relu = F.relu(x)
    x_path1 = module.path_1(x_relu)
    x_path2 = module.path_2(x_relu[:, :, 1:, 1:])
    return module.final_path_bn(torch.cat([x_path1, x_path2], 1))


class FactorizeReduce(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(FactorizeReduce, self).__init__()
        _factorize_reduce_build(self, in_ch, out_ch)

    def forward(self, x):
        return _factorize_reduce_forward(self, x)


OPS = {
    'sepconv3x3': lambda in_ch, out_ch, stride, stem: BranchSeparables(in_ch, out_ch, 3, stride, 1, stem=stem),
    'sepconv5x5': lambda in_ch, out_ch, stride, stem: BranchSeparables(in_ch, out_ch, 5, stride, 2, stem=stem),
    'sepconv7x7': lambda in_ch, out_ch, stride, stem: BranchSeparables(in_ch, out_ch, 7, stride, 3, stem=stem),
    'avgpool3x3': lambda in_ch, out_ch, stride, stem: Pool('avg', in_ch, out_ch, stride),
    'maxpool3x3': lambda in_ch, out_ch, stride, stem: Pool('max', in_ch, out_ch, stride),
    'identity': lambda in_ch, out_ch, stride, stem: Identity(in_ch, out_ch, stride),
}


class Cell(nn.Module):
    def __init__(self, operations, hidden_indices, concat_indices,
                 in_ch, in_ch_prev, channels, reduction=False, reduction_p=False, stem=False,
                 differentiable=False):
        # Cell: cell_prev -> cell_in -> cell_out
        # Filters: in_ch_prev -> in_ch -> out_ch
        # If reduction_p, cell_prev will have twice resolution as cell_in
        # If reduction, cell_out will have half resolution of cell_in
        super(Cell, self).__init__()
        self.operations = operations
        self.hidden_indices = hidden_indices
        self.concat_indices = concat_indices
        self.in_ch = in_ch
        self.in_ch_prev = in_ch_prev
        self.reduction = reduction
        self.reduction_p = reduction_p
        self.n_hidden = len(self.operations) // 2
        self.stem = stem
        self.differentiable = differentiable
        assert len(self.operations) == 2 * self.n_hidden
        assert len(self.hidden_indices) == 2 * self.n_hidden

        if self.differentiable:
            self.inputs = nn.ModuleList([SimpleDifferentiableTensorChoice(i // 2 + 2)
                                         for i in range(self.n_hidden * 2)])

        if isinstance(channels, int):
            self.channels = [channels] * (self.n_hidden + 2)
        else:
            self.channels = channels
        out_ch = self.channels[-1]
        if reduction_p:
            _factorize_reduce_build(self, in_ch_prev, out_ch)
        else:
            self.conv_prev_1x1 = StdConv(in_ch_prev, out_ch)
        self.conv_1x1 = StdConv(in_ch, out_ch)

        for i in range(self.n_hidden):
            i_left, i_right = self.hidden_indices[i * 2], self.hidden_indices[i * 2 + 1]
            compute_stride = lambda idx: 2 if reduction and idx < 2 else 1
            if self.differentiable:
                self.add_module(f'comb_iter_{i}_left',
                                nn.ModuleList([self._build_mixed_op(self.channels[i_left], out_ch, compute_stride(i_left), stem)
                                               for i_left in range(i + 2)]))
                self.add_module(f'comb_iter_{i}_right',
                                nn.ModuleList([self._build_mixed_op(self.channels[i_right], out_ch, compute_stride(i_right), stem)
                                               for i_right in range(i + 2)]))
            else:
                self.add_module(f'comb_iter_{i}_left',
                                OPS[self.operations[i * 2]](self.channels[i_left], out_ch, compute_stride(i_left), stem))
                self.add_module(f'comb_iter_{i}_right',
                                OPS[self.operations[i * 2 + 1]](self.channels[i_right], out_ch, compute_stride(i_right), stem))

        for i in range(2):
            if reduction and i in self.concat_indices:
                self.add_module(f'reduction_{i}', FactorizeReduce(self.channels[i], out_ch))

    def _build_mixed_op(self, in_ch, out_ch, stride, stem):
        return SimpleDifferentiableOpChoice([op(in_ch, out_ch, stride, stem) for op in OPS.values()])

    def cell(self, x, x_prev):
        # All the states (including x and x_prev) should have same filters.
        # In case of reduction, other states will have half resolution of the first two.
        # self.channels marks the filters each of the states owns.
        states = [x, x_prev]
        for i in range(self.n_hidden):
            left_op, right_op = getattr(self, f'comb_iter_{i}_left'), getattr(self, f'comb_iter_{i}_right')
            if self.differentiable:
                x_left = self.inputs[i * 2]([left_op[left_in](states[left_in]) for left_in in range(i + 2)])
                x_right = self.inputs[i * 2 + 1]([right_op[right_in](states[right_in]) for right_in in range(i + 2)])
            else:
                x_left = left_op(states[self.hidden_indices[i * 2]])
                x_right = right_op(states[self.hidden_indices[i * 2 + 1]])
            states.append(x_left + x_right)
        for i in range(2):
            if self.reduction and i in self.concat_indices:
                states[i] = getattr(self, f'reduction_{i}')(states[i])
        return torch.cat([states[i] for i in self.concat_indices], 1)

    def forward(self, x_prev, x=None):
        # To workaround with current IR translator that doesn't support ordering of arguments
        if self.stem:
            x = x_prev
            x1 = self.conv_1x1(x)
            out = self.cell(x1, x)
        else:
            if self.reduction_p:
                x_prev = _factorize_reduce_forward(self, x_prev)
            else:
                x_prev = self.conv_prev_1x1(x_prev)
            x = self.conv_1x1(x)
            out = self.cell(x, x_prev)
        return out


class CellStem0(nn.Module):
    def __init__(self, operations, hidden_indices, concat_indices, in_ch, channels):
        super().__init__()
        n_hidden = len(operations) // 2
        self.wrapper = Cell(operations, hidden_indices, concat_indices,
                            in_ch, in_ch, [channels, in_ch] + [channels] * n_hidden,
                            reduction=True, stem=True)

    def forward(self, x):
        return self.wrapper(x)


class CellStem1(nn.Module):
    def __init__(self, operations, hidden_indices, concat_indices, in_ch, in_ch_prev, channels):
        super().__init__()
        self.wrapper = Cell(operations, hidden_indices, concat_indices,
                            in_ch, in_ch_prev, channels, reduction=True, reduction_p=True)

    def forward(self, x_prev, x):
        return self.wrapper(x_prev, x)


class NASNet(nn.Module):
    def __init__(self, num_stem_features, num_normal_cells, filters, scaling, skip_reduction,
                 use_aux=True, num_classes=1000):
        super(NASNet, self).__init__()
        self.num_normal_cells = num_normal_cells
        self.skip_reduction = skip_reduction
        self.use_aux = use_aux
        self.num_classes = num_classes
        expand_normal = len(_NASNET_A_NORMAL['concat_indices'])
        expand_reduce = len(_NASNET_A_REDUCE['concat_indices'])

        self.conv0 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(3, num_stem_features, kernel_size=3, stride=2, bias=False)),
            ('bn', nn.BatchNorm2d(num_stem_features, eps=0.001, momentum=0.1, affine=True))
        ]))

        filters_p = int(filters * scaling ** (-1))
        filters_pp = int(filters * scaling ** (-2))
        self.cell_stem_0 = CellStem0(_NASNET_A_REDUCE['operations'], _NASNET_A_REDUCE['hidden_indices'],
                                     _NASNET_A_REDUCE['concat_indices'], num_stem_features, filters_pp)
        filters_pp *= expand_reduce
        self.cell_stem_1 = CellStem1(_NASNET_A_REDUCE['operations'], _NASNET_A_REDUCE['hidden_indices'],
                                     _NASNET_A_REDUCE['concat_indices'], filters_pp, num_stem_features, filters_p)
        filters_p *= expand_reduce

        cell_id = 0
        for i in range(3):
            self.add_module(f'cell_{cell_id}', Cell(_NASNET_A_NORMAL['operations'], _NASNET_A_NORMAL['hidden_indices'],
                                                    _NASNET_A_NORMAL['concat_indices'], filters_p,
                                                    filters_pp, filters, reduction_p=True))
            cell_id += 1
            filters_pp, filters_p = filters_p, expand_normal * filters
            for _ in range(num_normal_cells - 1):
                self.add_module(f'cell_{cell_id}', Cell(_NASNET_A_NORMAL['operations'], _NASNET_A_NORMAL['hidden_indices'],
                                                        _NASNET_A_NORMAL['concat_indices'], filters_p,
                                                        filters_pp, filters))
                filters_pp = filters_p
                cell_id += 1

            filters *= scaling
            if i < 2:
                self.add_module(f'reduction_cell_{i}',
                                Cell(_NASNET_A_REDUCE['operations'], _NASNET_A_REDUCE['hidden_indices'],
                                     _NASNET_A_REDUCE['concat_indices'], filters_p, filters_pp,
                                     filters, reduction=True))
                filters_p = expand_reduce * filters

        self.linear = nn.Linear(filters_p, self.num_classes)  # large: 4032; mobile: 1056

    def features(self, x):
        x_conv0 = self.conv0(x)
        x_stem_0 = self.cell_stem_0(x_conv0)
        x_stem_1 = self.cell_stem_1(x_conv0, x_stem_0)
        prev_x, x = x_stem_0, x_stem_1
        cell_id = 0
        for i in range(3):
            for _ in range(self.num_normal_cells):
                new_x = getattr(self, f'cell_{cell_id}')(prev_x, x)
                prev_x, x = x, new_x
                cell_id += 1
            if i < 2:
                new_x = getattr(self, f'reduction_cell_{i}')(prev_x, x)
                prev_x = x if not self.skip_reduction else prev_x
                x = new_x
        return x

    def logits(self, features):
        x = F.relu(features, inplace=False)
        x = F.avg_pool2d(x, kernel_size=int(x.size(2))).view(x.size(0), -1)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        output = self.logits(x)
        return output


def nasnet_a_mobile(num_classes=1000):
    return NASNet(32, 4, 44, 2, skip_reduction=False, use_aux=True, num_classes=num_classes)


def nasnet_a_large(num_classes=1000):
    return NASNet(96, 6, 168, 2, skip_reduction=True, use_aux=True, num_classes=num_classes)


class CellMutator(Mutator):
    def __init__(self, targets: 'List[str]', num_nodes: int, operation_candidates: 'List[str]',
                 differentiable: bool = False):
        self.targets = targets
        self.num_nodes = num_nodes
        self.operation_candidates = operation_candidates
        self.differentiable = differentiable

    def retrieve_targeted_graphs(self, graph: 'Graph') -> 'List[Graph]':
        return [graph.find_node(t) for t in self.targets]

    def _infer_output_channels(self, graph, node):
        if node.operation.type == 'Cell':
            return sum([c for i, c in enumerate(node.operation.params['channels'])
                        if i in node.operation.params['concat_indices']])
        if node.operation.type == 'BatchNorm2d':
            return node.operation.params['num_features']
        if node.operation.type == 'aten::add':
            channels = [self._infer_output_channels(graph, n) for n in graph.get_predecessors(node)]
            assert all([c == channels[0] for c in channels])
            return channels[0]
        if node.operation.type == 'aten::cat':
            channels = [self._infer_output_channels(graph, n) for n in graph.get_predecessors(node)]
            return sum(channels)
        raise AssertionError('Unsupported operation %s' % node.operation)

    def _update_successor_in_dim(self, graph, target_node, out_channels):
        for node in graph.get_successors(target_node):
            if node.operation.type in ('aten::relu', 'aten::avg_pool2d', 'aten::view', 'aten::dropout'):
                self._update_successor_in_dim(graph, node, out_channels)
            elif node.operation.type == 'Linear':
                node.update_operation(None, in_features=out_channels)

    def mutate(self, graph):
        if self.differentiable:
            normal_operations = [None] * (self.num_nodes * 2)
            normal_hidden_indices = [None] * (self.num_nodes * 2)
            normal_concat_indices = [i for i in range(2, 2 + self.num_nodes)]
        else:
            normal_operations = [self.choice(self.operation_candidates) for _ in range(self.num_nodes * 2)]
            normal_hidden_indices = [self.choice(list(range(i // 2 + 2))) for i in range(self.num_nodes * 2)]
            normal_concat_indices = [i for i in range(2, self.num_nodes + 2) if i not in normal_hidden_indices]
        if self.differentiable:
            reduce_operations = [None] * (self.num_nodes * 2)
            reduce_hidden_indices = [None] * (self.num_nodes * 2)
            reduce_concat_indices = [i for i in range(2, 2 + self.num_nodes)]
        else:
            reduce_operations = [self.choice(self.operation_candidates) for _ in range(self.num_nodes * 2)]
            reduce_hidden_indices = [self.choice(list(range(i // 2 + 2))) for i in range(self.num_nodes * 2)]
            reduce_concat_indices = [i for i in range(2, self.num_nodes + 2) if i not in reduce_hidden_indices]
        for target_node in self.retrieve_targeted_graphs(graph):
            predecessors = graph.get_predecessors(target_node)
            reduction = target_node.operation.params['reduction']
            if len(predecessors) == 2:
                p_previous, previous = graph.get_predecessors(target_node)
                in_ch = self._infer_output_channels(graph, previous)
                in_ch_prev = self._infer_output_channels(graph, p_previous)
            else:
                in_ch = in_ch_prev = self._infer_output_channels(graph, predecessors[0])
            target_node.update_operation(None, **{
                'operations': reduce_operations if reduction else normal_operations,
                'hidden_indices': reduce_hidden_indices if reduction else normal_hidden_indices,
                'concat_indices': reduce_concat_indices if reduction else normal_concat_indices,
                'in_ch': in_ch,
                'in_ch_prev': in_ch_prev,
                'differentiable': self.differentiable
            })
            self._update_successor_in_dim(graph, target_node, self._infer_output_channels(graph, target_node))


def nasnet():
    model = nasnet_a_mobile()
    cells = []
    for name, module in model.named_modules():
        if isinstance(module, Cell):
            cells.append(name)
    model_graph = gen_pytorch_graph(model, dummy_input=torch.randn(1, 3, 224, 224),
                                    collapsed_nodes={name: 'Cell' for name in cells})
    mutators = [CellMutator(cells, num_nodes=5, operation_candidates=list(OPS.keys()))]
    return model_graph, mutators


amoebanet = nasnet
pnas = nasnet


def darts():
    model = NASNet(32, 2, 16, 2, skip_reduction=False, use_aux=True, num_classes=10)
    cells = []
    for name, module in model.named_modules():
        if isinstance(module, Cell):
            cells.append(name)
    model_graph = gen_pytorch_graph(model, dummy_input=torch.randn(1, 3, 32, 32),
                                    collapsed_nodes={name: 'Cell' for name in cells})
    graph = Graph.load(model_graph)
    mutator = CellMutator(cells, num_nodes=5, operation_candidates=list(OPS.keys()), differentiable=True)
    super_graph = mutator.apply(graph, None)
    return super_graph.dump()
