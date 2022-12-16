# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from typing import Any, Callable, Dict

import torch
import torch.nn.functional as F

from ..base.compressor import Distiller
from ..base.target_space import DistillationTargetSpace
from ..utils.evaluator import Evaluator

_TARGET_SPACES = Dict[str, Dict[str, DistillationTargetSpace]]


class BasicDistiller(Distiller):
    def __init__(self, model: torch.nn.Module, evaluator: Evaluator, teacher_predict: Callable[[Any], torch.Tensor]):
        config_list = [{'op_names': [''], 'lambda': 1.0, 'apply_method': 'kl'}]
        super().__init__(model, config_list, evaluator=evaluator)
        self._target_spaces: _TARGET_SPACES
        self.teacher_predict = teacher_predict
        self.patch_loss()

    def patch_loss(self):
        def loss_patch(origin_loss, batch):
            t_out = self.teacher_predict(batch)
            new_loss = 0
            for _, ts in self._target_spaces.items():
                for _, target_space in ts.items():
                    hs = target_space.hidden_state
                    if target_space.apply_method == 'mse':
                        new_loss += target_space.lambda_ * F.mse_loss(hs, t_out)
                    else:
                        new_loss += target_space.lambda_ * F.kl_div(hs.log_softmax(dim=-1), t_out.softmax(dim=-1), reduction='batchmean')
                    target_space.clean()
            return origin_loss + new_loss
        self.evaluator.patch_loss(loss_patch)

    def compress(self, max_steps: int | None = None, max_epochs: int | None = None):
        self.evaluator.bind_model(self.bound_model)
        self.evaluator.train(max_steps=max_steps, max_epochs=max_epochs)
        self.evaluator.unbind_model()
