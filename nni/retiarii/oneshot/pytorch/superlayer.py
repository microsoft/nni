
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch.nn as nn
import torch.nn.functional as F

from nni.retiarii.nn.pytorch.api import ValueChoice

# SuperLinear(nn.Linear) 放一个单独文件
class PathSamplingSuperLinear(nn.Module):
    def __init__(self, module) -> None:
        super().__init__()
        args = module.trace_kwargs
        self.in_features = args['in_features']
        max_in_features = self.in_features
        if isinstance(self.in_features, ValueChoice):
            self.in_label = self.in_features.label
            max_in_features = max(self.in_features.candidates)

        self.out_features = args['out_features']
        max_out_features = self.out_features
        if isinstance(self.out_features, ValueChoice):
            self.out_label = self.out_features.label
            max_out_features = max(self.out_features.candidates)

        bias = args['bias'] if 'bias' in args.keys() else True

        device = args['device'] if 'device' in args.keys() else None

        dtype = args['dtype']  if 'dtype' in args.keys() else None

        self.fc = nn.Linear(max_in_features, max_out_features, bias, device, dtype)

    def forward(self, x):
        # 如果是 valuechoice 就去读 sample 的值，否则就是固定值
        in_dim = self.in_features.candidates[self.in_features.sampled] \
             if isinstance(self.in_features, ValueChoice) else self.in_features
        out_dim = self.out_features.candidates[self.out_features.sampled] \
             if isinstance(self.out_features, ValueChoice) else self.out_features

        weights = self.fc.weight[:out_dim, :in_dim]
        bias = self.fc.bias[:out_dim]

        return F.linear(x, weights, bias)



    # ENAS 与 random 共用
    # Conv2D valueChoice
    