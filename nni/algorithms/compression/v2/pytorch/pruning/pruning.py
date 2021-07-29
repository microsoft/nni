# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List

from torch import Tensor
from torch.nn import Module

from nni.algorithms.compression.v2.pytorch.common.consistent_task_generator import AGPTaskGenerator
from .basic_pruner import LevelPruner


class AGPPruning:
    def __init__(self, total_iteration: int, model: Module, config_list: List[Dict],
                 origin_masks: Dict[str, Dict[str, Tensor]] = {}, log_dir: str = './log') -> None:
        pass
