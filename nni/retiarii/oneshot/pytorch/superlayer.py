
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from turtle import forward
import torch.nn as nn
import torch.nn.functional as F

from nni.retiarii.nn.pytorch.api import ValueChoice

class ToSample:
    """
    Base class for all things to be sampled.
    """
    def __init__(self, label, candidates = [], sampled = 0) -> None:
        self.label = label
        self.candidates = candidates
        self.sampled = sampled
    
    @property
    def sampled_item(self):
        return self.candidates[self.sampled]
    
    def __len__(self):
        return len(self.candidates)


class ENASLinearValueChoice(ToSample):
    """
    """
    def __init__(self, value_choice):
        super().__init__(value_choice.label, value_choice.candidates)
        self.n_chosen = 1
    # ENAS 与 random 共用
    # Conv2D valueChoice

class RandomLinearValueChoice(ToSample):
    def __init__(self, value_choice):
        super().__init__(value_choice.label, value_choice.candidates)

# SuperLinear(nn.Linear) 放一个单独文件
class PathSamplingSuperLinear(nn.Linear):
    def __init__(self, module) -> None:
        args = module.trace_kwargs
        self.in_features = args['in_features']
        max_in_features =  max(self.in_features.candidates) if isinstance(self.in_features, ToSample) else self.in_features
        
        self.out_features = args['out_features']
        max_out_features = max(self.out_features.candidates) if isinstance(self.out_features, ToSample) else self.out_features

        bias = args['bias'] if 'bias' in args.keys() else True

        device = args['device'] if 'device' in args.keys() else None

        dtype = args['dtype']  if 'dtype' in args.keys() else None

        super().__init__(max_in_features, max_out_features, bias, device, dtype)

    def forward(self, x):
        # 如果是 valuechoice 就去读 sample 的值，否则就是固定值
        in_dim = self.in_features.sampled_item if isinstance(self.in_features, ToSample) else self.in_features
        out_dim = self.out_features.sampled_item if isinstance(self.out_features, ToSample) else self.out_features

        weights = self.weight[:out_dim, :in_dim]
        bias = self.bias[:out_dim]

        return F.linear(x, weights, bias)

class PathSamplingSuperConv2d(nn.Conv2d):
    def __init__(self, module) -> None:
        super().__init__()
        args = module.trace_kwargs
        self.in_features = args['in_features']

        super().__init__()
    
    def forward(self, x):
        
        pass

    