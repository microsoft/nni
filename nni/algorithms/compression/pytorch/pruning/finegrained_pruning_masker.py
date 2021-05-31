# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import torch
from .weight_masker import WeightMasker

__all__ = ['LevelPrunerMasker']

logger = logging.getLogger('torch pruner')


class LevelPrunerMasker(WeightMasker):
    """
    Prune to an exact pruning level specification
    """

    def calc_mask(self, sparsity, wrapper, wrapper_idx=None):
        weight = wrapper.module.weight.data.clone()
        if wrapper.weight_mask is not None:
            # apply base mask for iterative pruning
            weight = weight * wrapper.weight_mask

        w_abs = weight.abs()
        k = int(weight.numel() * sparsity)
        if k == 0:
            return {'weight_mask': torch.ones(weight.shape).type_as(weight)}
        threshold = torch.topk(w_abs.view(-1), k, largest=False)[0].max()
        mask_weight = torch.gt(w_abs, threshold).type_as(weight)
        mask = {'weight_mask': mask_weight}
        return mask
