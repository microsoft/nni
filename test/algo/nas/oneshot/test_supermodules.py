import pytest

import numpy as np
import torch
import torch.nn as nn

import nni
from nni.mutable import MutableList, frozen
from nni.nas.nn.pytorch import LayerChoice, ModelSpace, Cell, MutableConv2d, MutableBatchNorm2d, MutableLayerNorm, MutableLinear, MutableMultiheadAttention
from nni.nas.oneshot.pytorch.differentiable import DartsLightningModule
from nni.nas.oneshot.pytorch.strategy import RandomOneShot, DARTS
from nni.nas.oneshot.pytorch.supermodule.base import BaseSuperNetModule
from nni.nas.oneshot.pytorch.supermodule.differentiable import (
    MixedOpDifferentiablePolicy, DifferentiableMixedLayer, DifferentiableMixedInput, GumbelSoftmax,
    DifferentiableMixedRepeat, DifferentiableMixedCell
)
from nni.nas.oneshot.pytorch.supermodule.sampling import (
    MixedOpPathSamplingPolicy, PathSamplingLayer, PathSamplingInput, PathSamplingRepeat, PathSamplingCell
)
from nni.nas.oneshot.pytorch.supermodule.operation import MixedConv2d, NATIVE_MIXED_OPERATIONS
from nni.nas.oneshot.pytorch.supermodule.proxyless import ProxylessMixedLayer, ProxylessMixedInput
from nni.nas.oneshot.pytorch.supermodule._operation_utils import Slicable as S, MaybeWeighted as W
from nni.nas.oneshot.pytorch.supermodule._expression_utils import *

from ut.nas.nn.models import (
    CellSimple, CellDefaultArgs, CellCustomProcessor, CellLooseEnd, CellOpFactory
)

@pytest.fixture(autouse=True)
def context():
    frozen._ENSURE_FROZEN_STRICT = False
    yield
    frozen._ENSURE_FROZEN_STRICT = True


def test_slice():
    weight = np.ones((3, 7, 24, 23))
    assert S(weight)[:, 1:3, :, 9:13].shape == (3, 2, 24, 4)
    assert S(weight)[:, 1:W(3)*2+1, :, 9:13].shape == (3, 6, 24, 4)
    assert S(weight)[:, 1:W(3)*2+1].shape == (3, 6, 24, 23)

    # Ellipsis
    assert S(weight)[..., 9:13].shape == (3, 7, 24, 4)
    assert S(weight)[:2, ..., 1:W(3)+1].shape == (2, 7, 24, 3)
    assert S(weight)[..., 1:W(3)*2+1].shape == (3, 7, 24, 6)
    assert S(weight)[..., :10, 1:W(3)*2+1].shape == (3, 7, 10, 6)

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
    vc0 = nni.choice('exp', [3, 4, 6]) * 2 + nni.choice('add', [0, 1])

    assert vc0.freeze(chosen) == 7
    vc = vc0 + nni.choice('exp', [3, 4, 6])
    assert vc.freeze(chosen) == 10

    assert list(MutableList([vc0, vc]).simplify().keys()) == ['exp', 'add']

    assert traverse_all_options(vc) == [9, 10, 12, 13, 18, 19]
    weights = dict(traverse_all_options(vc, weights={'exp': [0.5, 0.3, 0.2], 'add': [0.4, 0.6]}))
    ans = dict([(9, 0.2), (10, 0.3), (12, 0.12), (13, 0.18), (18, 0.08), (19, 0.12)])
    assert len(weights) == len(ans)
    for value, weight in ans.items():
        assert abs(weight - weights[value]) < 1e-6

    assert evaluate_constant(nni.choice('x', [3, 4, 6]) - nni.choice('x', [3, 4, 6])) == 0
    with pytest.raises(ValueError):
        evaluate_constant(nni.choice('x', [3, 4, 6]) - nni.choice('y', [3, 4, 6]))

    assert evaluate_constant(nni.choice('x', [3, 4, 6]) * 2 / nni.choice('x', [3, 4, 6])) == 2


def test_expectation():
    vc = nni.choice('exp', [3, 4, 6]) * 2 + nni.choice('add', [0, 1])
    assert expression_expectation(vc, {'exp': [0.5, 0.3, 0.2], 'add': [0.4, 0.6]}) == 8.4

    vc = sum([nni.choice(f'e{i}', [0, 1]) for i in range(100)])
    assert expression_expectation(vc, {f'e{i}': [0.5] * 2 for i in range(100)}) == 50

    vc = nni.choice('a', [1, 2, 3]) * nni.choice('b', [1, 2, 3]) - nni.choice('c', [1, 2, 3])
    probs1 = [0.2, 0.3, 0.5]
    probs2 = [0.1, 0.2, 0.7]
    probs3 = [0.3, 0.4, 0.3]
    expect = sum(
        (i * j - k) * p1 * p2 * p3
        for i, p1 in enumerate(probs1, 1)
        for j, p2 in enumerate(probs2, 1)
        for k, p3 in enumerate(probs3, 1)
    )
    assert abs(expression_expectation(vc, {'a': probs1, 'b': probs2, 'c': probs3}) - expect) < 1e-12

    vc = nni.choice('a', [1, 2, 3]) + 1
    assert expression_expectation(vc, {'a': [0.2, 0.3, 0.5]}) == 3.3


def test_weighted_sum():
    weights = [0.1, 0.2, 0.7]
    items = [1, 2, 3]
    assert abs(weighted_sum(items, weights) - 2.6) < 1e-6

    assert weighted_sum(items) == 6

    with pytest.raises(TypeError, match='Unsupported'):
        weighted_sum(['a', 'b', 'c'], weights)

    assert abs(weighted_sum(np.arange(3), weights).item() - 1.6) < 1e-6

    items = [torch.full((2, 3, 5), i) for i in items]
    assert abs(weighted_sum(items, weights).flatten()[0].item() - 2.6) < 1e-6

    items = [torch.randn(2, 3, i) for i in [1, 2, 3]]
    with pytest.raises(ValueError, match=r'does not match.*\n.*torch\.Tensor\(2, 3, 1\)'):
        weighted_sum(items, weights)

    items = [(1, 2), (3, 4), (5, 6)]
    res = weighted_sum(items, weights)
    assert len(res) == 2 and abs(res[0] - 4.2) < 1e-6 and abs(res[1] - 5.2) < 1e-6

    items = [(1, 2), (3, 4), (5, 6, 7)]
    with pytest.raises(ValueError):
        weighted_sum(items, weights)

    items = [{"a": i, "b": np.full((2, 3, 5), i)} for i in [1, 2, 3]]
    res = weighted_sum(items, weights)
    assert res['b'].shape == (2, 3, 5)
    assert abs(res['b'][0][0][0] - res['a']) < 1e-6
    assert abs(res['a'] - 2.6) < 1e-6


def test_pathsampling_valuechoice():
    orig_conv = MutableConv2d(3, nni.choice('123', [3, 5, 7]), kernel_size=3)
    conv = MixedConv2d.mutate(orig_conv, 'dummy', {}, {'mixed_op_sampling': MixedOpPathSamplingPolicy})
    conv.resample(memo={'123': 5})
    assert conv(torch.zeros((1, 3, 5, 5))).size(1) == 5
    conv.resample(memo={'123': 7})
    assert conv(torch.zeros((1, 3, 5, 5))).size(1) == 7
    assert conv.export({})['123'] in [3, 5, 7]


def test_differentiable_valuechoice():
    orig_conv = MutableConv2d(3, nni.choice('456', [3, 5, 7]), 
        kernel_size=nni.choice('123', [3, 5, 7]),
        padding=nni.choice('123', [3, 5, 7]) // 2
    )
    memo = {
        '123': nn.Parameter(torch.zeros(3)),
        '456': nn.Parameter(torch.zeros(3)),
    }
    conv = MixedConv2d.mutate(orig_conv, 'dummy', memo, {'mixed_op_sampling': MixedOpDifferentiablePolicy})
    assert conv(torch.zeros((1, 3, 7, 7))).size(2) == 7

    assert set(conv.export({}).keys()) == {'123', '456'}


def test_differentiable_layerchoice_dedup():
    layerchoice1 = LayerChoice([MutableConv2d(3, 3, 3), MutableConv2d(3, 3, 3)], label='a')
    layerchoice2 = LayerChoice([MutableConv2d(3, 3, 3), MutableConv2d(3, 3, 3)], label='a')

    memo = {'a': nn.Parameter(torch.zeros(2))}
    DifferentiableMixedLayer.mutate(layerchoice1, 'x', memo, {})
    DifferentiableMixedLayer.mutate(layerchoice2, 'x', memo, {})
    assert len(memo) == 1 and 'a' in memo


def _mutate_op_path_sampling_policy(operation):
    for native_op in NATIVE_MIXED_OPERATIONS:
        if native_op.bound_type == type(operation):
            mutate_op = native_op.mutate(operation, 'dummy', {}, {'mixed_op_sampling': MixedOpPathSamplingPolicy})
            break
    return mutate_op


def _mixed_operation_sampling_sanity_check(operation, memo, *input):
    mutate_op = _mutate_op_path_sampling_policy(operation)
    mutate_op.resample(memo=memo)
    return mutate_op(*input)


def _mixed_operation_state_dict_sanity_check(operation, model, memo, *input):
    mutate_op = _mutate_op_path_sampling_policy(operation)
    mutate_op.resample(memo=memo)
    frozen_op = mutate_op.freeze(memo)
    return frozen_op(*input), mutate_op(*input)


def _mixed_operation_differentiable_sanity_check(operation, *input):
    memo = {k: nn.Parameter(torch.zeros(len(v))) for k, v in operation.simplify().items()}
    for native_op in NATIVE_MIXED_OPERATIONS:
        if native_op.bound_type == type(operation):
            mutate_op = native_op.mutate(operation, 'dummy', memo, {'mixed_op_sampling': MixedOpDifferentiablePolicy})
            break

    mutate_op(*input)
    mutate_op.export({})
    mutate_op.export_probs({})


def test_mixed_linear():
    linear = MutableLinear(nni.choice('shared', [3, 6, 9]), nni.choice('xx', [2, 4, 8]))
    _mixed_operation_sampling_sanity_check(linear, {'shared': 3}, torch.randn(2, 3))
    _mixed_operation_sampling_sanity_check(linear, {'shared': 9}, torch.randn(2, 9))
    _mixed_operation_differentiable_sanity_check(linear, torch.randn(2, 9))

    linear = MutableLinear(nni.choice('shared', [3, 6, 9]), nni.choice('xx', [2, 4, 8]), bias=False)
    _mixed_operation_sampling_sanity_check(linear, {'shared': 3}, torch.randn(2, 3))

    with pytest.raises(TypeError):
        linear = MutableLinear(nni.choice('shared', [3, 6, 9]), nni.choice('xx', [2, 4, 8]), bias=nni.choice('yy', [False, True]))
        _mixed_operation_sampling_sanity_check(linear, {'shared': 3}, torch.randn(2, 3))

    linear = MutableLinear(nni.choice('in_features', [3, 6, 9]), nni.choice('out_features', [2, 4, 8]), bias=True)
    kwargs = {'in_features': 6, 'out_features': 4}
    out1, out2 = _mixed_operation_state_dict_sanity_check(linear, MutableLinear(**kwargs), kwargs, torch.randn(2, 6))
    assert torch.allclose(out1, out2)


def test_mixed_conv2d():
    conv = MutableConv2d(nni.choice('in', [3, 6, 9]), nni.choice('out', [2, 4, 8]) * 2, 1)
    assert _mixed_operation_sampling_sanity_check(conv, {'in': 3, 'out': 4}, torch.randn(2, 3, 9, 9)).size(1) == 8
    _mixed_operation_differentiable_sanity_check(conv, torch.randn(2, 9, 3, 3))

    # stride
    conv = MutableConv2d(nni.choice('in', [3, 6, 9]), nni.choice('out', [2, 4, 8]), 1, stride=nni.choice('stride', [1, 2]))
    assert _mixed_operation_sampling_sanity_check(conv, {'in': 3, 'stride': 2}, torch.randn(2, 3, 10, 10)).size(2) == 5
    assert _mixed_operation_sampling_sanity_check(conv, {'in': 3, 'stride': 1}, torch.randn(2, 3, 10, 10)).size(2) == 10
    with pytest.raises(ValueError, match='must not be mutable'):
        _mixed_operation_differentiable_sanity_check(conv, torch.randn(2, 9, 10, 10))

    # groups, dw conv
    conv = MutableConv2d(nni.choice('in', [3, 6, 9]), nni.choice('in', [3, 6, 9]), 1, groups=nni.choice('in', [3, 6, 9]))
    assert _mixed_operation_sampling_sanity_check(conv, {'in': 6}, torch.randn(2, 6, 10, 10)).size() == torch.Size([2, 6, 10, 10])

    # groups, invalid case
    conv = MutableConv2d(nni.choice('in', [9, 6, 3]), nni.choice('in', [9, 6, 3]), 1, groups=9)
    with pytest.raises(RuntimeError):
        assert _mixed_operation_sampling_sanity_check(conv, {'in': 6}, torch.randn(2, 6, 10, 10))

    # groups, differentiable
    conv = MutableConv2d(nni.choice('in', [3, 6, 9]), nni.choice('out', [3, 6, 9]), 1, groups=nni.choice('in', [3, 6, 9]))
    _mixed_operation_differentiable_sanity_check(conv, torch.randn(2, 9, 3, 3))

    conv = MutableConv2d(nni.choice('in', [3, 6, 9]), nni.choice('in', [3, 6, 9]), 1, groups=nni.choice('in', [3, 6, 9]))
    _mixed_operation_differentiable_sanity_check(conv, torch.randn(2, 9, 3, 3))

    with pytest.raises(ValueError):
        conv = MutableConv2d(nni.choice('in', [3, 6, 9]), nni.choice('in', [3, 6, 9]), 1, groups=nni.choice('groups', [3, 9]))
        _mixed_operation_differentiable_sanity_check(conv, torch.randn(2, 9, 3, 3))

    with pytest.raises(RuntimeError):
        conv = MutableConv2d(nni.choice('in', [3, 6, 9]), nni.choice('in', [3, 6, 9]), 1, groups=nni.choice('in', [3, 6, 9]) // 3)
        _mixed_operation_differentiable_sanity_check(conv, torch.randn(2, 10, 3, 3))

    # make sure kernel is sliced correctly
    conv = MutableConv2d(1, 1, nni.choice('k', [1, 3]), bias=False)
    conv = MixedConv2d.mutate(conv, 'dummy', {}, {'mixed_op_sampling': MixedOpPathSamplingPolicy})
    with torch.no_grad():
        conv.weight.zero_()
        # only center is 1, must pick center to pass this test
        conv.weight[0, 0, 1, 1] = 1
    conv.resample({'k': 1})
    assert conv(torch.ones((1, 1, 3, 3))).sum().item() == 9

    # only `in_channels`, `out_channels`, `kernel_size`, and `groups` influence state_dict
    conv = MutableConv2d(
        nni.choice('in_channels', [2, 4, 8]), nni.choice('out_channels', [6, 12, 24]), 
        kernel_size=nni.choice('kernel_size', [3, 5, 7]), groups=nni.choice('groups', [1, 2])   
    )
    kwargs = {
        'in_channels': 8, 'out_channels': 12, 
        'kernel_size': 5, 'groups': 2
    }
    out1, out2 = _mixed_operation_state_dict_sanity_check(conv, MutableConv2d(**kwargs), kwargs, torch.randn(2, 8, 16, 16))
    assert torch.allclose(out1, out2)

def test_mixed_batchnorm2d():
    bn = MutableBatchNorm2d(nni.choice('dim', [32, 64]))

    assert _mixed_operation_sampling_sanity_check(bn, {'dim': 32}, torch.randn(2, 32, 3, 3)).size(1) == 32
    assert _mixed_operation_sampling_sanity_check(bn, {'dim': 64}, torch.randn(2, 64, 3, 3)).size(1) == 64

    _mixed_operation_differentiable_sanity_check(bn, torch.randn(2, 64, 3, 3))

    bn = MutableBatchNorm2d(nni.choice('num_features', [32, 48, 64]))
    kwargs = {'num_features': 48}
    out1, out2 = _mixed_operation_state_dict_sanity_check(bn, MutableBatchNorm2d(**kwargs), kwargs, torch.randn(2, 48, 3, 3))
    assert torch.allclose(out1, out2)

def test_mixed_layernorm():
    ln = MutableLayerNorm(nni.choice('normalized_shape', [32, 64]), elementwise_affine=True)

    assert _mixed_operation_sampling_sanity_check(ln, {'normalized_shape': 32}, torch.randn(2, 16, 32)).size(-1) == 32
    assert _mixed_operation_sampling_sanity_check(ln, {'normalized_shape': 64}, torch.randn(2, 16, 64)).size(-1) == 64

    _mixed_operation_differentiable_sanity_check(ln, torch.randn(2, 16, 64))
    
    import itertools
    ln = MutableLayerNorm(nni.choice('normalized_shape', list(itertools.product([16, 32, 64], [8, 16]))))

    assert list(_mixed_operation_sampling_sanity_check(ln, {'normalized_shape': (16, 8)}, torch.randn(2, 16, 8)).shape[-2:]) == [16, 8]
    assert list(_mixed_operation_sampling_sanity_check(ln, {'normalized_shape': (64, 16)}, torch.randn(2, 64, 16)).shape[-2:]) == [64, 16]

    _mixed_operation_differentiable_sanity_check(ln, torch.randn(2, 64, 16))

    ln = MutableLayerNorm(nni.choice('normalized_shape', [32, 48, 64]))
    kwargs = {'normalized_shape': 48}
    out1, out2 = _mixed_operation_state_dict_sanity_check(ln, MutableLayerNorm(**kwargs), kwargs, torch.randn(2, 8, 48))
    assert torch.allclose(out1, out2)

def test_mixed_mhattn():
    mhattn = MutableMultiheadAttention(nni.choice('emb', [4, 8]), 4)

    assert _mixed_operation_sampling_sanity_check(mhattn, {'emb': 4},
        torch.randn(7, 2, 4), torch.randn(7, 2, 4), torch.randn(7, 2, 4))[0].size(-1) == 4
    assert _mixed_operation_sampling_sanity_check(mhattn, {'emb': 8},
        torch.randn(7, 2, 8), torch.randn(7, 2, 8), torch.randn(7, 2, 8))[0].size(-1) == 8

    _mixed_operation_differentiable_sanity_check(mhattn, torch.randn(7, 2, 8), torch.randn(7, 2, 8), torch.randn(7, 2, 8))

    mhattn = MutableMultiheadAttention(nni.choice('emb', [4, 8]), nni.choice('heads', [2, 3, 4]))
    assert _mixed_operation_sampling_sanity_check(mhattn, {'emb': 4, 'heads': 2},
        torch.randn(7, 2, 4), torch.randn(7, 2, 4), torch.randn(7, 2, 4))[0].size(-1) == 4
    with pytest.raises(AssertionError, match='divisible'):
        assert _mixed_operation_sampling_sanity_check(mhattn, {'emb': 4, 'heads': 3},
            torch.randn(7, 2, 4), torch.randn(7, 2, 4), torch.randn(7, 2, 4))[0].size(-1) == 4

    mhattn = MutableMultiheadAttention(nni.choice('emb', [4, 8]), 4, kdim=nni.choice('kdim', [5, 7]))
    assert _mixed_operation_sampling_sanity_check(mhattn, {'emb': 4, 'kdim': 7},
        torch.randn(7, 2, 4), torch.randn(7, 2, 7), torch.randn(7, 2, 4))[0].size(-1) == 4
    assert _mixed_operation_sampling_sanity_check(mhattn, {'emb': 8, 'kdim': 5},
        torch.randn(7, 2, 8), torch.randn(7, 2, 5), torch.randn(7, 2, 8))[0].size(-1) == 8

    mhattn = MutableMultiheadAttention(nni.choice('emb', [4, 8]), 4, vdim=nni.choice('vdim', [5, 8]))
    assert _mixed_operation_sampling_sanity_check(mhattn, {'emb': 4, 'vdim': 8},
        torch.randn(7, 2, 4), torch.randn(7, 2, 4), torch.randn(7, 2, 8))[0].size(-1) == 4
    assert _mixed_operation_sampling_sanity_check(mhattn, {'emb': 8, 'vdim': 5},
        torch.randn(7, 2, 8), torch.randn(7, 2, 8), torch.randn(7, 2, 5))[0].size(-1) == 8

    _mixed_operation_differentiable_sanity_check(mhattn, torch.randn(5, 3, 8), torch.randn(5, 3, 8), torch.randn(5, 3, 8))

    mhattn = MutableMultiheadAttention(embed_dim=nni.choice('embed_dim', [4, 8, 16]), num_heads=nni.choice('num_heads', [1, 2, 4]),
        kdim=nni.choice('kdim', [4, 8, 16]), vdim=nni.choice('vdim', [4, 8, 16]))
    kwargs = {'embed_dim': 16, 'num_heads': 2, 'kdim': 4, 'vdim': 8}
    (out1, _), (out2, _) = _mixed_operation_state_dict_sanity_check(mhattn, MutableMultiheadAttention(**kwargs), kwargs, torch.randn(7, 2, 16), torch.randn(7, 2, 4), torch.randn(7, 2, 8))
    assert torch.allclose(out1, out2)

@pytest.mark.skipif(torch.__version__.startswith('1.7'), reason='batch_first is not supported for legacy PyTorch')
def test_mixed_mhattn_batch_first():
    # batch_first is not supported for legacy pytorch versions
    # mark 1.7 because 1.7 is used on legacy pipeline

    mhattn = MutableMultiheadAttention(nni.choice('emb', [4, 8]), 2, kdim=(nni.choice('kdim', [3, 7])), vdim=nni.choice('vdim', [5, 8]),
                                bias=False, add_bias_kv=True, batch_first=True)
    assert _mixed_operation_sampling_sanity_check(mhattn, {'emb': 4, 'kdim': 7, 'vdim': 8},
        torch.randn(2, 7, 4), torch.randn(2, 7, 7), torch.randn(2, 7, 8))[0].size(-1) == 4
    assert _mixed_operation_sampling_sanity_check(mhattn, {'emb': 8, 'kdim': 3, 'vdim': 5},
        torch.randn(2, 7, 8), torch.randn(2, 7, 3), torch.randn(2, 7, 5))[0].size(-1) == 8

    _mixed_operation_differentiable_sanity_check(mhattn, torch.randn(1, 7, 8), torch.randn(1, 7, 7), torch.randn(1, 7, 8))


def test_pathsampling_layer_input():
    op = PathSamplingLayer({'a': MutableLinear(2, 3, bias=False), 'b': MutableLinear(2, 3, bias=True)}, label='ccc')
    with pytest.raises(RuntimeError, match='sample'):
        op(torch.randn(4, 2))

    op.resample({})
    assert op(torch.randn(4, 2)).size(-1) == 3
    assert op.simplify()['ccc'].values == ['a', 'b']
    assert op.export({})['ccc'] in ['a', 'b']

    input = PathSamplingInput(5, 2, 'concat', 'ddd')
    sample = input.resample({})
    assert 'ddd' in sample
    assert len(sample['ddd']) == 2
    assert input([torch.randn(4, 2) for _ in range(5)]).size(-1) == 4
    assert len(input.export({})['ddd']) == 2


def test_differentiable_layer_input():
    op = DifferentiableMixedLayer({'a': MutableLinear(2, 3, bias=False), 'b': MutableLinear(2, 3, bias=True)}, nn.Parameter(torch.randn(2)), nn.Softmax(-1), 'eee')
    assert op(torch.randn(4, 2)).size(-1) == 3
    assert op.export({})['eee'] in ['a', 'b']
    probs = op.export_probs({})
    assert len(probs) == 1
    assert len(probs['eee']) == 2
    assert abs(probs['eee']['a'] + probs['eee']['b'] - 1) < 1e-4
    assert len(list(op.parameters())) == 4
    assert len(list(op.arch_parameters())) == 1

    with pytest.raises(ValueError):
        op = DifferentiableMixedLayer({'a': MutableLinear(2, 3), 'b': MutableLinear(2, 4)}, nn.Parameter(torch.randn(2)), nn.Softmax(-1), 'eee')
        op(torch.randn(4, 2))

    input = DifferentiableMixedInput(5, 2, nn.Parameter(torch.zeros(5)), GumbelSoftmax(-1), 'ddd')
    assert input([torch.randn(4, 2) for _ in range(5)]).size(-1) == 2
    assert len(input.export({})['ddd']) == 2
    assert len(input.export_probs({})) == 1
    assert len(input.export_probs({})['ddd']) == 5
    assert 3 in input.export_probs({})['ddd']


def test_proxyless_layer_input():
    op = ProxylessMixedLayer({'a': MutableLinear(2, 3, bias=False), 'b': MutableLinear(2, 3, bias=True)}, nn.Parameter(torch.randn(2)),
                             nn.Softmax(-1), 'eee')
    assert op.resample({})['eee'] in ['a', 'b']
    assert op(torch.randn(4, 2)).size(-1) == 3
    assert op.export({})['eee'] in ['a', 'b']
    assert len(list(op.parameters())) == 4
    assert len(list(op.arch_parameters())) == 1

    input = ProxylessMixedInput(5, 2, nn.Parameter(torch.zeros(5)), GumbelSoftmax(-1), 'ddd')
    assert all(x in list(range(5)) for x in input.resample({})['ddd'])
    assert input([torch.randn(4, 2) for _ in range(5)]).size() == torch.Size([4, 2])
    exported = input.export({})['ddd']
    assert len(exported) == 2 and all(e in list(range(5)) for e in exported)


def test_pathsampling_repeat():
    op = PathSamplingRepeat([MutableLinear(16, 16), MutableLinear(16, 8), MutableLinear(8, 4)], nni.choice('ccc', [1, 2, 3]))
    sample = op.resample({})
    assert sample['ccc'] in [1, 2, 3]
    for i in range(1, 4):
        op.resample({'ccc': i})
        out = op(torch.randn(2, 16))
        assert out.shape[1] == [16, 8, 4][i - 1]

    op = PathSamplingRepeat([MutableLinear(i + 1, i + 2) for i in range(7)], 2 * nni.choice('ddd', [1, 2, 3]) + 1)
    sample = op.resample({})
    assert sample['ddd'] in [1, 2, 3]
    for i in range(1, 4):
        op.resample({'ddd': i})
        out = op(torch.randn(2, 1))
        assert out.shape[1] == (2 * i + 1) + 1


def test_differentiable_repeat():
    op = DifferentiableMixedRepeat(
        [MutableLinear(8 if i == 0 else 16, 16) for i in range(4)],
        nni.choice('ccc', [0, 1]) * 2 + 1,
        GumbelSoftmax(-1),
        {'ccc': nn.Parameter(torch.randn(2))},
    )
    op.resample({})
    assert op(torch.randn(2, 8)).size() == torch.Size([2, 16])
    sample = op.export({})
    assert 'ccc' in sample and sample['ccc'] in [0, 1]
    assert sorted(op.export_probs({})['ccc'].keys()) == [0, 1]

    class TupleModule(nn.Module):
        def __init__(self, num):
            super().__init__()
            self.num = num

        def forward(self, *args, **kwargs):
            return torch.full((2, 3), self.num), torch.full((3, 5), self.num), {'a': 7, 'b': [self.num] * 11}

    class CustomSoftmax(nn.Softmax):
        def forward(self, *args, **kwargs):
            return [0.3, 0.3, 0.4]

    op = DifferentiableMixedRepeat(
        [TupleModule(i + 1) for i in range(4)],
        nni.choice('ccc', [1, 2, 4]),
        CustomSoftmax(),
        {'ccc': nn.Parameter(torch.randn(3))},
    )
    op.resample({})
    res = op(None)
    assert len(res) == 3
    assert res[0].shape == (2, 3) and res[0][0][0].item() == 2.5
    assert res[2]['a'] == 7
    assert len(res[2]['b']) == 11 and res[2]['b'][-1] == 2.5


def test_pathsampling_cell():
    for cell_cls in [CellSimple, CellDefaultArgs, CellCustomProcessor, CellLooseEnd, CellOpFactory]:
        model = cell_cls()
        strategy = RandomOneShot()
        model = strategy.mutate_model(model)
        nas_modules = [m for m in model.modules() if isinstance(m, BaseSuperNetModule)]
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
        strategy = DARTS()
        model = strategy.mutate_model(model)
        nas_modules = [m for m in model.modules() if isinstance(m, BaseSuperNetModule)]
        result = {}
        for module in nas_modules:
            result.update(module.export(memo=result))
        assert len(result) == model.cell.num_nodes * model.cell.num_ops_per_node * 2
        for k, v in result.items():
            if 'input' in k:
                assert isinstance(v, list) and len(v) == 1

        result_prob = {}
        for module in nas_modules:
            result_prob.update(module.export_probs(memo=result_prob))

        ctrl_params = []
        for m in nas_modules:
            ctrl_params += list(m.arch_parameters())
        if cell_cls in [CellLooseEnd, CellOpFactory]:
            assert len(ctrl_params) == model.cell.num_nodes * (model.cell.num_nodes + 3) // 2
            assert len(result_prob) == len(ctrl_params)  # len(op_names) == 2
            for v in result_prob.values():
                assert len(v) == 2
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


def test_memo_sharing():
    class TestModelSpace(ModelSpace):
        def __init__(self):
            super().__init__()
            self.linear1 = Cell(
                [nn.Linear(16, 16), nn.Linear(16, 16, bias=False)],
                num_nodes=3, num_ops_per_node=2, num_predecessors=2, merge_op='loose_end',
                label='cell'
            )
            self.linear2 = Cell(
                [nn.Linear(16, 16), nn.Linear(16, 16, bias=False)],
                num_nodes=3, num_ops_per_node=2, num_predecessors=2, merge_op='loose_end',
                label='cell'
            )

    strategy = DARTS()
    model = strategy.mutate_model(TestModelSpace())
    assert model.linear1._arch_alpha['cell/2_0'] is model.linear2._arch_alpha['cell/2_0']


def test_parameters():
    class Model(ModelSpace):
        def __init__(self):
            super().__init__()
            self.op = DifferentiableMixedLayer(
                {
                    'a': MutableLinear(2, 3, bias=False),
                    'b': MutableLinear(2, 3, bias=True)
                },
                nn.Parameter(torch.randn(2)), nn.Softmax(-1), 'abc'
            )

        def forward(self, x):
            return self.op(x)

    model = Model()
    assert len(list(model.parameters())) == 4
    assert len(list(model.op.arch_parameters())) == 1

    optimizer = torch.optim.SGD(model.parameters(), 0.1)
    assert len(DartsLightningModule(model).arch_parameters()) == 1
    optimizer = DartsLightningModule(model).postprocess_weight_optimizers(optimizer)
    assert len(optimizer.param_groups[0]['params']) == 3
