# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from nni.common.version import torch_version_is_2

if torch_version_is_2():
    from torch.optim.lr_scheduler import LRScheduler  # type: ignore
    SCHEDULER = LRScheduler
else:
    from torch.optim.lr_scheduler import _LRScheduler
    SCHEDULER = _LRScheduler
