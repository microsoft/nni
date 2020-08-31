# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import torch
import torch.nn as nn
from ..compressor import Pruner
from .one_shot import _Constrained_StructuredFilterPruner
from ..speedup.compressor import get_module_by_name

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

class SELayer(nn.Module):
    """
    SE Layer from the paper "Squeeze-and-Excitation Networks".
    """
    def __init__(self, channel, device, reduction=16, record_weights=False):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1).to(device)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
            # nn.Softmax()
        ).to(device)
        # if record the weights of the channels during the forward.
        self.record_weights = record_weights
        self.sum_weights = torch.zeros(channel).to(device)
        self.count = 0

    def forward(self, x):
        n, c, _, _ = x.size()
        y = self.avg_pool(x).view(n, c)
        weights = self.fc(y)
        if self.record_weights:
            with torch.no_grad():
                # print(self.sum_weights.size())
                # print(weights.data.size())
                # print(weights.data)
                # print(self.sum_weights)
                # exit(-1)
                self.count += 1
                self.sum_weights += weights.transpose(1, 0).mean(1)
        # print(weights.size())
        # print(weights)
        y = weights.view(n, c, 1, 1)
        result = x * y.expand_as(x)
        return result


class AttentionActivationPruner(_Constrained_StructuredFilterPruner):
    """
    One-shot pruner based on the attention mechaism.
    """
    def __init__(self, model, config_list, dummy_input, optimizer, evaluator, finetuner, reduction=16):
        """
        Parameters
        ----------

        Returns
        -------
        """
        self.bound_model = model
        super(AttentionActivationPruner, self).__init__(model, config_list, dummy_input, pruning_algorithm='attention', optimizer=None)
        self._unwrap_model() # unwrap the model
        self.dummy_input =  dummy_input
        self.evaluator = evaluator
        self.finetuner = finetuner
        self.config_list = config_list
        self.reduction = reduction
        self.optimizer = optimizer
        # save the original state_dict for the model
        self.optimizer_state = optimizer.state_dict()

    def compress(self):
        layer_infors = self._detect_modules_to_compress()
        name_to_compress = [layer[0].name for layer in layer_infors]
        # freeze the original weighs of the model
        _logger.info('Freeze the original weights of the model')
        # for para in self.bound_model.parameters():
        #     para.requires_grad = False
        se_blocks = {}
        ori_convs = {}
        # for name, module in self.bound_model.named_modules():
        #     def new_forward(func, name):
        #         def forward(*args):
        #             print()
        #             print('In forward name', name)
        #             print('Input size', args[0].size())
        #             print('input id', id(args[0]))
        #             result =  func(*args)
        #             print('output id', id(result))
        #             return result
        #         return forward
        #     setattr(module, 'forward', new_forward(module.forward, name))
        for name in name_to_compress: 
            father_mod, son_mod = get_module_by_name(self.bound_model, name)
            errmsg = "Currently, only support the pruning for Conv layers, %s is not a conv layer"  % name
            assert isinstance(son_mod, nn.Conv2d) or isinstance(son_mod, nn.Conv1d), errmsg
            # father_mod.origin_conv = son_mod # save the original conv layer
            ori_convs[name] = son_mod
            # dummy_input is on the same device with the model
            se_layer = SELayer(son_mod.out_channels, self.dummy_input.device)
            se_layer.name = name
            new_mod = nn.Sequential(son_mod, se_layer)
            se_blocks[name] = se_layer
            # use the conv + attention block to replace the original conv layers
            setattr(father_mod, name.split('.')[-1], new_mod)
            _logger.info('Add SE block for %s', name)

        # finetune the attention block parameters
        _logger.info('Finetuning the weights of SEBlock')
        self.finetuner(self.bound_model)

        # also freeze the weight for the se layers
        for name in se_blocks:
            # start recording the weights of the channel before
            # evaluation
            se_blocks[name].record_weights = True
            # freeze the attention weights
            for para in se_blocks[name].parameters():
                para.requires_grad = False
        _logger.info('Calculaing the attention weights for each channel')
        self.evaluator(self.bound_model)
        # reset the model
        for name in name_to_compress:
            father_mod, son_mod = get_module_by_name(self.bound_model, name)
            # save the attention weights in the conv layer
            # print('sum_weights')
            print(name)
            print(se_blocks[name].sum_weights)
            # print(se_blocks[name].sum_weights.size())
            ori_convs[name].attention_weights = se_blocks[name].sum_weights / se_blocks[name].count
            setattr(father_mod, name.split('.')[-1], ori_convs[name])

        self._wrap_model()
        # unfreeze the weights
        # for para in self.bound_model.parameters():
        #     para.requires_grad = True
        # Calculate the mask based on the attention weights
        self.update_mask()
        return self.bound_model
