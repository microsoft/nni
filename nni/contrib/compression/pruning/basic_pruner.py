from typing import Dict

import torch

from ..base.compressor import Pruner


class BasicPruner(Pruner):
    def _collect_data(self) -> Dict[str, Dict[str, torch.Tensor]]:
        
