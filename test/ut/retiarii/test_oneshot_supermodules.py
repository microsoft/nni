import pytest

import numpy as np
import torch
import torch.nn as nn
from nni.retiarii.nn.pytorch import ValueChoice, Conv2d, BatchNorm2d, Linear, MultiheadAttention
from nni.retiarii.oneshot.pytorch.base_lightning import traverse_and_mutate_submodules
from nni.retiarii.oneshot.pytorch.supermodule.differentiable import (
    MixedOpDifferentiablePolicy, DifferentiableMixedLayer, DifferentiableMixedInput, GumbelSoftmax,
    DifferentiableMixedRepeat, DifferentiableMixedCell
)
from nni.retiarii.oneshot.pytorch.supermodule.sampling import (
    MixedOpPathSamplingPolicy, PathSamplingLayer, PathSamplingInput, PathSamplingRepeat, PathSamplingCell
)
from nni.retiarii.oneshot.pytorch.supermodule.operation import MixedConv2d, NATIVE_MIXED_OPERATIONS
from nni.retiarii.oneshot.pytorch.supermodule.proxyless import ProxylessMixedLayer, ProxylessMixedInput
from nni.retiarii.oneshot.pytorch.supermodule._operation_utils import Slicable as S, MaybeWeighted as W
from nni.retiarii.oneshot.pytorch.supermodule._valuechoice_utils import *

from .models import (
    CellSimple, CellDefaultArgs, CellCustomProcessor, CellLooseEnd, CellOpFactory
)


def test_slice():
    weight = np.ones((3, 7, 24, 23))
    assert S(weight)[:, 1:3, :, 9:13].shape == (3, 2, 24, 4)
    assert S(weight)[:, 1:W(3)*2+1, :, 9:13].shape == (3, 6, 24, 4)
    assert S(weight)[:, 1:W(3)*2+1].shape == (3, 6, 24, 23)

    # no effect
    assert S(weight)[:] is weight

    # list
    assert S(weight)[[slice(1), slice(2, 3)]].shape == (2, 7, 24, 23)
    assert S(weight)[[slice(1), slice(2, W(2) + 1)], W(2):].shape == (2, 5, 24, 23)

    # weighted
    weight = S(weight)[:W({1: 0.5, 2: 0.3, 3: 0.2})]
    weight = weight[:, 0, 0, 0]
    assert weight[0] == 1 and weight[1] == 0.5 and weight[2] == 0.2

    weight = np.ones((3, 6, 6))
    value = W({1: 0.5, 3: 0.5})
    weight = S(weight)[:, 3 - value:3 + value, 3 - value:3 + value]
    for i in range(0, 6):
        for j in range(0, 6):
            if 2 <= i <= 3 and 2 <= j <= 3:
                assert weight[0, i, j] == 1
            else:
                assert weight[1, i, j] == 0.5

    # weighted + list
    value = W({1: 0.5, 3: 0.5})
    weight = np.ones((8, 4))
    weight = S(weight)[[slice(value), slice(4, value + 4)]]
    assert weight.sum(1).tolist() == [4, 2, 2, 0, 4, 2, 2, 0]

    with pytest.raises(ValueError, match='one distinct'):
        # has to be exactly the same instance, equal is not enough
        weight = S(weight)[:W({1: 0.5}), : W({1: 0.5})]


def test_valuechoice_utils():
    chosen = {"exp": 3, "add": 1}
    vc0 = ValueChoice([3, 4, 6], label='exp') * 2 + ValueChoice([0, 1], label='add')

    assert evaluate_value_choice_with_dict(vc0, chosen) == 7
    vc = vc0 + ValueChoice([3, 4, 6], label='exp')
    assert evaluate_value_choice_with_dict(vc, chosen) == 10

    assert list(dedup_inner_choices([vc0, vc]).keys()) == ['exp', 'add']

    assert traverse_all_options(vc) == [9, 10, 12, 13, 18, 19]
    weights = dict(traverse_all_options(vc, weights={'exp': [0.5, 0.3, 0.2], 'add': [0.4, 0.6]}))
    ans = dict([(9, 0.2), (10, 0.3), (12, 0.12), (13, 0.18), (18, 0.08), (19, 0.12)])
    assert len(weights) == len(ans)
    for value, weight in ans.items():
        assert abs(weight - weights[value]) < 1e-6


def test_pathsampling_valuechoice():
    orig_conv = Conv2d(3, ValueChoice([3, 5, 7], label='123'), kernel_size=3)
    conv = MixedConv2d.mutate(orig_conv, 'dummy', {}, {'mixed_op_sampling': MixedOpPathSamplingPolicy})
    conv.resample(memo={'123': 5})
    assert conv(torch.zeros((1, 3, 5, 5))).size(1) == 5
    conv.resample(memo={'123': 7})
    assert conv(torch.zeros((1, 3, 5, 5))).size(1) == 7
    assert conv.export({})['123'] in [3, 5, 7]


def test_differentiable_valuechoice():
    orig_conv = Conv2d(3, ValueChoice([3, 5, 7], label='456'), kernel_size=ValueChoice(
        [3, 5, 7], label='123'), padding=ValueChoice([3, 5, 7], label='123') // 2)
    conv = MixedConv2d.mutate(orig_conv, 'dummy', {}, {'mixed_op_sampling': MixedOpDifferentiablePolicy})
    assert conv(torch.zeros((1, 3, 7, 7))).size(2) == 7

    assert set(conv.export({}).keys()) == {'123', '456'}


def _mixed_operation_sampling_sanity_check(operation, memo, *input):
    for native_op in NATIVE_MIXED_OPERATIONS:
        if native_op.bound_type == type(operation):
            mutate_op = native_op.mutate(operation, 'dummy', {}, {'mixed_op_sampling': MixedOpPathSamplingPolicy})
            break

    mutate_op.resample(memo=memo)
    return mutate_op(*input)


def _mixed_operation_differentiable_sanity_check(operation, *input):
    for native_op in NATIVE_MIXED_OPERATIONS:
        if native_op.bound_type == type(operation):
            mutate_op = native_op.mutate(operation, 'dummy', {}, {'mixed_op_sampling': MixedOpDifferentiablePolicy})
            break

    return mutate_op(*input)


def test_mixed_linear():
    linear = Linear(ValueChoice([3, 6, 9], label='shared'), ValueChoice([2, 4, 8]))
    _mixed_operation_sampling_sanity_check(linear, {'shared': 3}, torch.randn(2, 3))
    _mixed_operation_sampling_sanity_check(linear, {'shared': 9}, torch.randn(2, 9))
    _mixed_operation_differentiable_sanity_check(linear, torch.randn(2, 9))

    linear = Linear(ValueChoice([3, 6, 9], label='shared'), ValueChoice([2, 4, 8]), bias=False)
    _mixed_operation_sampling_sanity_check(linear, {'shared': 3}, torch.randn(2, 3))

    with pytest.raises(TypeError):
        linear = Linear(ValueChoice([3, 6, 9], label='shared'), ValueChoice([2, 4, 8]), bias=ValueChoice([False, True]))
        _mixed_operation_sampling_sanity_check(linear, {'shared': 3}, torch.randn(2, 3))


def test_mixed_conv2d():
    conv = Conv2d(ValueChoice([3, 6, 9], label='in'), ValueChoice([2, 4, 8], label='out') * 2, 1)
    assert _mixed_operation_sampling_sanity_check(conv, {'in': 3, 'out': 4}, torch.randn(2, 3, 9, 9)).size(1) == 8
    _mixed_operation_differentiable_sanity_check(conv, torch.randn(2, 9, 3, 3))

    # stride
    conv = Conv2d(ValueChoice([3, 6, 9], label='in'), ValueChoice([2, 4, 8], label='out'), 1, stride=ValueChoice([1, 2], label='stride'))
    assert _mixed_operation_sampling_sanity_check(conv, {'in': 3, 'stride': 2}, torch.randn(2, 3, 10, 10)).size(2) == 5
    assert _mixed_operation_sampling_sanity_check(conv, {'in': 3, 'stride': 1}, torch.randn(2, 3, 10, 10)).size(2) == 10

    # groups, dw conv
    conv = Conv2d(ValueChoice([3, 6, 9], label='in'), ValueChoice([3, 6, 9], label='in'), 1, groups=ValueChoice([3, 6, 9], label='in'))
    assert _mixed_operation_sampling_sanity_check(conv, {'in': 6}, torch.randn(2, 6, 10, 10)).size() == torch.Size([2, 6, 10, 10])

    # make sure kernel is sliced correctly
    conv = Conv2d(1, 1, ValueChoice([1, 3], label='k'), bias=False)
    conv = MixedConv2d.mutate(conv, 'dummy', {}, {'mixed_op_sampling': MixedOpPathSamplingPolicy})
    with torch.no_grad():
        conv.weight.zero_()
        # only center is 1, must pick center to pass this test
        conv.weight[0, 0, 1, 1] = 1
    conv.resample({'k': 1})
    assert conv(torch.ones((1, 1, 3, 3))).sum().item() == 9


def test_mixed_batchnorm2d():
    bn = BatchNorm2d(ValueChoice([32, 64], label='dim'))

    assert _mixed_operation_sampling_sanity_check(bn, {'dim': 32}, torch.randn(2, 32, 3, 3)).size(1) == 32
    assert _mixed_operation_sampling_sanity_check(bn, {'dim': 64}, torch.randn(2, 64, 3, 3)).size(1) == 64

    _mixed_operation_differentiable_sanity_check(bn, torch.randn(2, 64, 3, 3))


def test_mixed_mhattn():
    mhattn = MultiheadAttention(ValueChoice([4, 8], label='emb'), 4)

    assert _mixed_operation_sampling_sanity_check(mhattn, {'emb': 4},
        torch.randn(7, 2, 4), torch.randn(7, 2, 4), torch.randn(7, 2, 4))[0].size(-1) == 4
    assert _mixed_operation_sampling_sanity_check(mhattn, {'emb': 8},
        torch.randn(7, 2, 8), torch.randn(7, 2, 8), torch.randn(7, 2, 8))[0].size(-1) == 8

    _mixed_operation_differentiable_sanity_check(mhattn, torch.randn(7, 2, 8), torch.randn(7, 2, 8), torch.randn(7, 2, 8))

    mhattn = MultiheadAttention(ValueChoice([4, 8], label='emb'), ValueChoice([2, 3, 4], label='heads'))
    assert _mixed_operation_sampling_sanity_check(mhattn, {'emb': 4, 'heads': 2},
        torch.randn(7, 2, 4), torch.randn(7, 2, 4), torch.randn(7, 2, 4))[0].size(-1) == 4
    with pytest.raises(AssertionError, match='divisible'):
        assert _mixed_operation_sampling_sanity_check(mhattn, {'emb': 4, 'heads': 3},
            torch.randn(7, 2, 4), torch.randn(7, 2, 4), torch.randn(7, 2, 4))[0].size(-1) == 4

    mhattn = MultiheadAttention(ValueChoice([4, 8], label='emb'), 4, kdim=ValueChoice([5, 7], label='kdim'))
    assert _mixed_operation_sampling_sanity_check(mhattn, {'emb': 4, 'kdim': 7},
        torch.randn(7, 2, 4), torch.randn(7, 2, 7), torch.randn(7, 2, 4))[0].size(-1) == 4
    assert _mixed_operation_sampling_sanity_check(mhattn, {'emb': 8, 'kdim': 5},
        torch.randn(7, 2, 8), torch.randn(7, 2, 5), torch.randn(7, 2, 8))[0].size(-1) == 8

    mhattn = MultiheadAttention(ValueChoice([4, 8], label='emb'), 4, vdim=ValueChoice([5, 8], label='vdim'))
    assert _mixed_operation_sampling_sanity_check(mhattn, {'emb': 4, 'vdim': 8},
        torch.randn(7, 2, 4), torch.randn(7, 2, 4), torch.randn(7, 2, 8))[0].size(-1) == 4
    assert _mixed_operation_sampling_sanity_check(mhattn, {'emb': 8, 'vdim': 5},
        torch.randn(7, 2, 8), torch.randn(7, 2, 8), torch.randn(7, 2, 5))[0].size(-1) == 8

    _mixed_operation_differentiable_sanity_check(mhattn, torch.randn(5, 3, 8), torch.randn(5, 3, 8), torch.randn(5, 3, 8))


@pytest.mark.skipif(torch.__version__.startswith('1.7'), reason='batch_first is not supported for legacy PyTorch')
def test_mixed_mhattn_batch_first():
    # batch_first is not supported for legacy pytorch versions
    # mark 1.7 because 1.7 is used on legacy pipeline

    mhattn = MultiheadAttention(ValueChoice([4, 8], label='emb'), 2, kdim=(ValueChoice([3, 7], label='kdim')), vdim=ValueChoice([5, 8], label='vdim'),
                                bias=False, add_bias_kv=True, batch_first=True)
    assert _mixed_operation_sampling_sanity_check(mhattn, {'emb': 4, 'kdim': 7, 'vdim': 8},
        torch.randn(2, 7, 4), torch.randn(2, 7, 7), torch.randn(2, 7, 8))[0].size(-1) == 4
    assert _mixed_operation_sampling_sanity_check(mhattn, {'emb': 8, 'kdim': 3, 'vdim': 5},
        torch.randn(2, 7, 8), torch.randn(2, 7, 3), torch.randn(2, 7, 5))[0].size(-1) == 8

    _mixed_operation_differentiable_sanity_check(mhattn, torch.randn(1, 7, 8), torch.randn(1, 7, 7), torch.randn(1, 7, 8))


def test_pathsampling_layer_input():
    op = PathSamplingLayer([('a', Linear(2, 3, bias=False)), ('b', Linear(2, 3, bias=True))], label='ccc')
    with pytest.raises(RuntimeError, match='sample'):
        op(torch.randn(4, 2))

    op.resample({})
    assert op(torch.randn(4, 2)).size(-1) == 3
    assert op.search_space_spec()['ccc'].values == ['a', 'b']
    assert op.export({})['ccc'] in ['a', 'b']

    input = PathSamplingInput(5, 2, 'concat', 'ddd')
    sample = input.resample({})
    assert 'ddd' in sample
    assert len(sample['ddd']) == 2
    assert input([torch.randn(4, 2) for _ in range(5)]).size(-1) == 4
    assert len(input.export({})['ddd']) == 2


def test_differentiable_layer_input():
    op = DifferentiableMixedLayer([('a', Linear(2, 3, bias=False)), ('b', Linear(2, 3, bias=True))], nn.Parameter(torch.randn(2)), nn.Softmax(-1), 'eee')
    assert op(torch.randn(4, 2)).size(-1) == 3
    assert op.export({})['eee'] in ['a', 'b']
    assert len(list(op.parameters())) == 3

    input = DifferentiableMixedInput(5, 2, nn.Parameter(torch.zeros(5)), GumbelSoftmax(-1), 'ddd')
    assert input([torch.randn(4, 2) for _ in range(5)]).size(-1) == 2
    assert len(input.export({})['ddd']) == 2


def test_proxyless_layer_input():
    op = ProxylessMixedLayer([('a', Linear(2, 3, bias=False)), ('b', Linear(2, 3, bias=True))], nn.Parameter(torch.randn(2)), nn.Softmax(-1), 'eee')
    assert op.resample({})['eee'] in ['a', 'b']
    assert op(torch.randn(4, 2)).size(-1) == 3
    assert op.export({})['eee'] in ['a', 'b']
    assert len(list(op.parameters())) == 3

    input = ProxylessMixedInput(5, 2, nn.Parameter(torch.zeros(5)), GumbelSoftmax(-1), 'ddd')
    assert input.resample({})['ddd'] in list(range(5))
    assert input([torch.randn(4, 2) for _ in range(5)]).size() == torch.Size([4, 2])
    assert input.export({})['ddd'] in list(range(5))


def test_pathsampling_repeat():
    op = PathSamplingRepeat([nn.Linear(16, 16), nn.Linear(16, 8), nn.Linear(8, 4)], ValueChoice([1, 2, 3], label='ccc'))
    sample = op.resample({})
    assert sample['ccc'] in [1, 2, 3]
    for i in range(1, 4):
        op.resample({'ccc': i})
        out = op(torch.randn(2, 16))
        assert out.shape[1] == [16, 8, 4][i - 1]

    op = PathSamplingRepeat([nn.Linear(i + 1, i + 2) for i in range(7)], 2 * ValueChoice([1, 2, 3], label='ddd') + 1)
    sample = op.resample({})
    assert sample['ddd'] in [1, 2, 3]
    for i in range(1, 4):
        op.resample({'ddd': i})
        out = op(torch.randn(2, 1))
        assert out.shape[1] == (2 * i + 1) + 1


def test_differentiable_repeat():
    op = DifferentiableMixedRepeat(
        [nn.Linear(8 if i == 0 else 16, 16) for i in range(4)],
        ValueChoice([0, 1], label='ccc') * 2 + 1,
        GumbelSoftmax(-1),
        {}
    )
    op.resample({})
    assert op(torch.randn(2, 8)).size() == torch.Size([2, 16])
    sample = op.export({})
    assert 'ccc' in sample and sample['ccc'] in [0, 1]


def test_pathsampling_cell():
    for cell_cls in [CellSimple, CellDefaultArgs, CellCustomProcessor, CellLooseEnd, CellOpFactory]:
        model = cell_cls()
        nas_modules = traverse_and_mutate_submodules(model, [
            PathSamplingLayer.mutate,
            PathSamplingInput.mutate,
            PathSamplingCell.mutate,
        ], {})
        result = {}
        for module in nas_modules:
            result.update(module.resample(memo=result))
        assert len(result) == model.cell.num_nodes * model.cell.num_ops_per_node * 2
        result = {}
        for module in nas_modules:
            result.update(module.export(memo=result))
        assert len(result) == model.cell.num_nodes * model.cell.num_ops_per_node * 2

        if cell_cls in [CellLooseEnd, CellOpFactory]:
            assert isinstance(model.cell, PathSamplingCell)
        else:
            assert not isinstance(model.cell, PathSamplingCell)

        inputs = {
            CellSimple: (torch.randn(2, 16), torch.randn(2, 16)),
            CellDefaultArgs: (torch.randn(2, 16),),
            CellCustomProcessor: (torch.randn(2, 3), torch.randn(2, 16)),
            CellLooseEnd: (torch.randn(2, 16), torch.randn(2, 16)),
            CellOpFactory: (torch.randn(2, 3), torch.randn(2, 16)),
        }[cell_cls]

        output = model(*inputs)
        if cell_cls == CellCustomProcessor:
            assert isinstance(output, tuple) and len(output) == 2 and \
                output[1].shape == torch.Size([2, 16 * model.cell.num_nodes])
        else:
            # no loose-end support for now
            assert output.shape == torch.Size([2, 16 * model.cell.num_nodes])


def test_differentiable_cell():
    for cell_cls in [CellSimple, CellDefaultArgs, CellCustomProcessor, CellLooseEnd, CellOpFactory]:
        model = cell_cls()
        nas_modules = traverse_and_mutate_submodules(model, [
            DifferentiableMixedLayer.mutate,
            DifferentiableMixedInput.mutate,
            DifferentiableMixedCell.mutate,
        ], {})
        result = {}
        for module in nas_modules:
            result.update(module.export(memo=result))
        assert len(result) == model.cell.num_nodes * model.cell.num_ops_per_node * 2

        ctrl_params = []
        for m in nas_modules:
            ctrl_params += list(m.parameters(arch=True))
        if cell_cls in [CellLooseEnd, CellOpFactory]:
            assert len(ctrl_params) == model.cell.num_nodes * (model.cell.num_nodes + 3) // 2
            assert isinstance(model.cell, DifferentiableMixedCell)
        else:
            assert not isinstance(model.cell, DifferentiableMixedCell)

        inputs = {
            CellSimple: (torch.randn(2, 16), torch.randn(2, 16)),
            CellDefaultArgs: (torch.randn(2, 16),),
            CellCustomProcessor: (torch.randn(2, 3), torch.randn(2, 16)),
            CellLooseEnd: (torch.randn(2, 16), torch.randn(2, 16)),
            CellOpFactory: (torch.randn(2, 3), torch.randn(2, 16)),
        }[cell_cls]

        output = model(*inputs)
        if cell_cls == CellCustomProcessor:
            assert isinstance(output, tuple) and len(output) == 2 and \
                output[1].shape == torch.Size([2, 16 * model.cell.num_nodes])
        else:
            # no loose-end support for now
            assert output.shape == torch.Size([2, 16 * model.cell.num_nodes])

