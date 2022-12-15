from typing import Callable, Dict

import torch
import torch.nn.functional as F

from ..base.compressor import Distiller
from ..base.target_space import DistillationTargetSpace
from ..utils.evaluator import Evaluator

_TARGET_SPACES = Dict[str, Dict[str, DistillationTargetSpace]]


class BasicDistiller(Distiller):
    def __init__(self, model: torch.nn.Module, evaluator: Evaluator, teacher_predict: Callable):
        config_list = [{'op_names': [''], 'lambda': 1.0}]
        super().__init__(model, config_list, evaluator=evaluator)
        self._target_spaces: _TARGET_SPACES
        self.teacher_predict = teacher_predict

    def patch_loss(self):
        def loss_patch(origin_loss, batch):
            t_out = self.teacher_predict(batch)
            new_loss = 0
            for _, ts in self._target_spaces.items():
                for _, target_space in ts.items():
                    hs = target_space.hidden_state
                    if target_space.apply_method == 'mse':
                        new_loss += target_space.lambda_ * F.mse_loss(hs, t_out)
                        target_space.clean()
            return origin_loss + new_loss
        self.evaluator.patch_loss(loss_patch)
