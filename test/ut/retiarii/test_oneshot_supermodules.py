import torch
from nni.retiarii.nn.pytorch import ValueChoice
from nni.retiarii.oneshot.pytorch.supermodule.sampling import PathSamplingConv2d


def test_pathsampling_valuechoice():
    conv = PathSamplingConv2d({"in_channels": 3, "out_channels": ValueChoice([3, 5, 7], label='123'), "kernel_size": 3})
    conv.resample(memo={'123': 5})
    assert conv(torch.zeros((1, 3, 5, 5))).size(1) == 5
    conv.resample(memo={'123': 7})
    assert conv(torch.zeros((1, 3, 5, 5))).size(1) == 7


test_pathsampling_valuechoice()
