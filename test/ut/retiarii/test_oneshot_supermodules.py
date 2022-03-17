import pytest

import numpy as np
import torch
from nni.retiarii.nn.pytorch import ValueChoice, Conv2d
from nni.retiarii.oneshot.pytorch.supermodule.differentiable import DifferentiableMixedOperation
from nni.retiarii.oneshot.pytorch.supermodule.sampling import PathSamplingOperation
from nni.retiarii.oneshot.pytorch.supermodule.operation import MixedConv2d
from nni.retiarii.oneshot.pytorch.supermodule._operation_utils import Slicable as S, MaybeWeighted as W


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
