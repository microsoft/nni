import torch
from nni.retiarii.nn.pytorch import ValueChoice, Conv2d
from nni.retiarii.oneshot.pytorch.supermodule.differentiable import DifferentiableMixedOperation
from nni.retiarii.oneshot.pytorch.supermodule.sampling import PathSamplingOperation
from nni.retiarii.oneshot.pytorch.supermodule.operation import SuperConv2d


def test_pathsampling_valuechoice():
    orig_conv = Conv2d(3, ValueChoice([3, 5, 7], label='123'), kernel_size=3)
    conv = SuperConv2d.mutate(orig_conv, 'dummy', {}, {'mixed_op_sampling_strategy': PathSamplingOperation})
    conv.resample(memo={'123': 5})
    assert conv(torch.zeros((1, 3, 5, 5))).size(1) == 5
    conv.resample(memo={'123': 7})
    assert conv(torch.zeros((1, 3, 5, 5))).size(1) == 7
    assert conv.export({})['123'] in [3, 5, 7]


def test_differentiable_valuechoice():
    orig_conv = Conv2d(3, ValueChoice([3, 5, 7], label='456'), kernel_size=ValueChoice([3, 5, 7], label='123'), padding=ValueChoice([3, 5, 7], label='123') // 2)
    conv = SuperConv2d.mutate(orig_conv, 'dummy', {}, {'mixed_op_sampling_strategy': DifferentiableMixedOperation})
    assert conv(torch.zeros((1, 3, 7, 7))).size(2) == 7



test_pathsampling_valuechoice()
test_differentiable_valuechoice()
