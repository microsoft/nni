# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import torch
import torch.nn as nn
from ..compressor import Pruner, _Constrained_StructuredFilterPruner
from nni.compression.torch.speedup.compressor import get_module_by_name
_logger = logging.getLogger(__name__)

class SELayer(nn.Module):
    """
    SE Layer from the paper "Squeeze-and-Excitation Networks".
    """
    def __init__(self, channel, reduction=16, record_weights=False):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        # if record the weights of the channels during the forward.
        self.record_weights = record_weights
        self.sum_weights = torch.zeros(channel)
        self.count = 0

    def forward(self, x):
        n, c, _, _ = x.size()
        y = self.avg_pool(x).view(n, c)
        weights = self.fc(y)
        if self.record_weights:
            with torch.no_grad():
                self.count += 1
                self.sum_weights += weights.data
        y= weights.view(n, c, 1, 1)
        return x * y.expand_as(x)


class AttentionActivationPruner(_Constrained_StructuredFilterPruner):
    """
    One-shot pruner based on the attention mechaism.
    """
    def __init__(self, model, config_list, evaluator, finetuner, reduction=16):
        """
        Parameters
        ----------

        Returns
        -------
        """
        self.bound_model = model
        super(AttentionActivationPruner, self).__init__(model, config_list)
        self._unwrap_model() # unwrap the model
        self.evaluator = evaluator
        self.finetuner = finetuner
        self.config_list = config_list
        self.reduction = reduction

    def compress(self):
        need_prune = set()
        for cfg in self.config_list:
            for name in cfg['op_names']:
                need_prune.add(name)
        # freeze the original weighs of the model
        for para in self.bound_model.parameters():
            para.requires_grad = False
        se_blocks = {}
        for name, module in self.bound_model.named_modules():
            if name in need_prune:
                errmsg = "Currently, only support the pruning for Conv layers, %s is not a conv layer" %name
                assert isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d), errmsg
                father_mod, son_mod = get_module_by_name(self.bound_model, name)
                father_mod.origin_conv = son_mod # save the original conv layer
                se_layer = SELayer(son_mod.out_channels)
                new_mod = nn.Sequential(son_mod, se_layer)
                se_blocks[name] = se_layer
                # use the conv + attention block to replace the original conv layers
                setattr(father_mod, name.split('.')[-1], new_mod)
        # finetune the attention block parameters
        self.finetuner(self.bound_model)
        # also freeze the weight for the se layers
        for name in se_blocks:
            # start recording the weights of the channel before
            # evaluation
            se_blocks.records_weights = True
            for para in se_blocks[name].parameters():
                para.requires_grad = False
        self.evaluator(self.bound_model)
        # reset the model
        for name, module in self.bound_model.named_modules():
            if name in need_prune:
                father_mod, son_mod = get_module_by_name(self.bound_model, name)
                # save the attention weights in the conv layer
                son_mod.attention_weights = se_blocks[name].sum_weights / se_blocks[name].count
                setattr(father_mod, name.split('.')[-1], father_mod.origin_conv)
                delattr(father_mod, 'origin_conv')
        # unfreeze the weights
        for para in self.bound_model.parameters():
            para.requires_grad = True
        # Calculate the mask based on the attention weights
        self.update_mask()
        return self.bound_model
