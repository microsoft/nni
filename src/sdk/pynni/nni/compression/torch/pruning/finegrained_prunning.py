# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import torch
from .weight_masker import WeightMasker

__all__ = ['LevelPrunerMasker', 'SlimPrunerMasker']

logger = logging.getLogger('torch pruner')


class LevelPrunerMasker(WeightMasker):
    """
    Prune to an exact pruning level specification
    """

    def calc_mask(self, sparsity, wrapper, wrapper_idx):
        weight = wrapper.module.weight.data.clone()
        if wrapper.weight_mask is not None:
            weight = weight * wrapper.weight_mask

        w_abs = weight.abs()
        k = int(weight.numel() * sparsity)
        if k == 0:
            return {'weight_mask': torch.ones(weight.shape).type_as(weight)}
        threshold = torch.topk(w_abs.view(-1), k, largest=False)[0].max()
        mask_weight = torch.gt(w_abs, threshold).type_as(weight)
        mask = {'weight_mask': mask_weight}
        return mask

class SlimPrunerMasker(WeightMasker):
    """
    A structured pruning algorithm that prunes channels by pruning the weights of BN layers.
    Zhuang Liu, Jianguo Li, Zhiqiang Shen, Gao Huang, Shoumeng Yan and Changshui Zhang
    "Learning Efficient Convolutional Networks through Network Slimming", 2017 ICCV
    https://arxiv.org/pdf/1708.06519.pdf
    """

    def __init__(self, model, pruner, **kwargs):
        super().__init__(model, pruner)
        weight_list = []
        for (layer, _) in pruner.get_modules_to_compress():
            weight_list.append(layer.module.weight.data.abs().clone())
        all_bn_weights = torch.cat(weight_list)
        k = int(all_bn_weights.shape[0] * pruner.config_list[0]['sparsity'])
        self.global_threshold = torch.topk(all_bn_weights.view(-1), k, largest=False)[0].max()

    def calc_mask(self, sparsity, wrapper, wrapper_idx):
        assert wrapper.type == 'BatchNorm2d', 'SlimPruner only supports 2d batch normalization layer pruning'
        weight = wrapper.module.weight.data.clone()
        if wrapper.weight_mask is not None:
            weight = weight * wrapper.weight_mask

        base_mask = torch.ones(weight.size()).type_as(weight).detach()
        mask = {'weight_mask': base_mask.detach(), 'bias_mask': base_mask.clone().detach()}
        filters = weight.size(0)
        num_prune = int(filters * sparsity)
        if filters >= 2 and num_prune >= 1:
            w_abs = weight.abs()
            mask_weight = torch.gt(w_abs, self.global_threshold).type_as(weight)
            mask_bias = mask_weight.clone()
            mask = {'weight_mask': mask_weight.detach(), 'bias_mask': mask_bias.detach()}
        return mask
