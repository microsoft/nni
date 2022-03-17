import pytest

import numpy as np
import torch
from nni.retiarii.nn.pytorch import ValueChoice, Conv2d
from nni.retiarii.oneshot.pytorch.supermodule.differentiable import DifferentiableMixedOperation
from nni.retiarii.oneshot.pytorch.supermodule.sampling import PathSamplingOperation
from nni.retiarii.oneshot.pytorch.supermodule.operation import MixedConv2d
from nni.retiarii.oneshot.pytorch.supermodule._operation_utils import Slicable as S, MaybeWeighted as W
from nni.retiarii.oneshot.pytorch.supermodule._valuechoice_utils import *


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
    conv = MixedConv2d.mutate(orig_conv, 'dummy', {}, {'mixed_op_sampling_strategy': PathSamplingOperation})
    conv.resample(memo={'123': 5})
    assert conv(torch.zeros((1, 3, 5, 5))).size(1) == 5
    conv.resample(memo={'123': 7})
    assert conv(torch.zeros((1, 3, 5, 5))).size(1) == 7
    assert conv.export({})['123'] in [3, 5, 7]


def test_differentiable_valuechoice():
    orig_conv = Conv2d(3, ValueChoice([3, 5, 7], label='456'), kernel_size=ValueChoice(
        [3, 5, 7], label='123'), padding=ValueChoice([3, 5, 7], label='123') // 2)
    conv = MixedConv2d.mutate(orig_conv, 'dummy', {}, {'mixed_op_sampling_strategy': DifferentiableMixedOperation})
    assert conv(torch.zeros((1, 3, 7, 7))).size(2) == 7


test_pathsampling_valuechoice()
test_differentiable_valuechoice()
test_slice()
test_valuechoice_utils()
