# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

import torch
from nni.common.concrete_trace_utils import concrete_trace
from nni.compression.speedup.dependency import build_channel_dependency


# should have dependency between conv1 and conv2 and conv3 and conv4
class PatternA(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(3, 16, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(3, 16, 3, 1, 1)
        self.conv4 = torch.nn.Conv2d(3, 16, 3, 1, 1)
        
    def forward(self, x: torch.Tensor):
        return torch.cat((self.conv1(x), self.conv2(x)), dim = 0) + torch.cat((self.conv3(x), self.conv4(x)), dim = 0)

# should not have dependency
class PatternB(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(3, 32, 3, 1, 1)
        
    def forward(self, x: torch.Tensor):
        return torch.cat((self.conv1(x), self.conv2(x)), dim = 1)
    
# should not have dependency (shape breakpoint)
class PatternC(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(3, 16, 3, 1, 1)
        
    def forward(self, x: torch.Tensor):
        return torch.sum(self.conv1(x), dim = 1) + torch.sum(self.conv2(x), dim = 1)

def check_sets_all_match(a, b):
    assert len(a) == len(b)
    for s in a:
        s = {node.name for node in s}
        assert s in b
        b.pop(b.index(s))
    assert len(b) == 0

@pytest.mark.parametrize('mod, deps', [
    (PatternA, [{'conv1', 'conv2'}, {'conv3', 'conv4'}]),
    (PatternB, []),
    (PatternC, []),
])
def test_channel_dependency(mod, deps):
    model = mod()
    dummy_input = (torch.randn(1, 3, 224, 224), )
    traced = concrete_trace(model, dummy_input)
    dependency = build_channel_dependency(traced)
    check_sets_all_match(dependency, deps)
