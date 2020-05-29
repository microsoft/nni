# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import logging
import importlib
import torch
from schema import And, Optional
from .constants import masker_dict
from ..utils.config_validation import CompressorSchema
from ..compressor import Pruner

__all__ = ['LevelPruner', 'SlimPruner', 'L1FilterPruner', 'L2FilterPruner', 'FPGMPruner', 'TaylorFOWeightFilterPruner']

logger = logging.getLogger('torch pruner')

class OneshotPruner(Pruner):
    """
    Prune to an exact pruning level specification
    """

    def __init__(self, model, config_list, pruning_algorithm='level', optimizer=None):
        """
        Parameters
        ----------
        model : torch.nn.module
            Model to be pruned
        config_list : list
            List on pruning configs
        optimizer: torch.optim.Optimizer
            Optimizer used to train model
        """

        super().__init__(model, config_list, optimizer)
        self.set_wrappers_attribute("if_calculated", False)
        self.masker = masker_dict[pruning_algorithm](model, self)

    def validate_config(self, model, config_list):
        """
        Parameters
        ----------
        model : torch.nn.module
            Model to be pruned
        config_list : list
            List on pruning configs
        """
        schema = CompressorSchema([{
            'sparsity': And(float, lambda n: 0 < n < 1),
            Optional('op_types'): [str],
            Optional('op_names'): [str]
        }], model, logger)

        schema.validate(config_list)

    def calc_mask(self, wrapper, **kwargs):
        """
        Calculate the mask of given layer
        Parameters
        ----------
        wrapper : Module
            the module to instrument the compression operation

        Returns
        -------
        dict
            dictionary for storing masks
        """
        if wrapper.if_calculated:
            return None

        sparsity = wrapper.config['sparsity']
        weight = wrapper.module.weight.data
        bias = None
        if hasattr(wrapper.module, 'bias') and wrapper.module.bias is not None:
            bias = wrapper.module.bias.data
        if not wrapper.if_calculated:
            masks = self._do_calc_mask(weight, bias=bias, sparsity=sparsity, wrapper=wrapper)
            if masks is not None:
                wrapper.if_calculated = True
            return masks
        else:
            return None

    def _do_calc_mask(self, weight, bias=None, sparsity=1., wrapper=None):
        return self.masker.calc_mask(weight, bias=bias, sparsity=sparsity, wrapper=wrapper)


class LevelPruner(OneshotPruner):
    def __init__(self, model, config_list, optimizer=None):
        super().__init__(model, config_list, pruning_algorithm='level', optimizer=optimizer)

class SlimPruner(OneshotPruner):
    def __init__(self, model, config_list, optimizer=None):
        super().__init__(model, config_list, pruning_algorithm='slim', optimizer=optimizer)

    def validate_config(self, model, config_list):
        schema = CompressorSchema([{
            'sparsity': And(float, lambda n: 0 < n < 1),
            'op_types': ['BatchNorm2d'],
            Optional('op_names'): [str]
        }], model, logger)

        schema.validate(config_list)

class _StructuredFilterPruner(OneshotPruner):
    def __init__(self, model, config_list, pruning_algorithm, optimizer=None):
        super().__init__(model, config_list, pruning_algorithm=pruning_algorithm, optimizer=optimizer)

    def validate_config(self, model, config_list):
        schema = CompressorSchema([{
            'sparsity': And(float, lambda n: 0 < n < 1),
            'op_types': ['Conv2d'],
            Optional('op_names'): [str]
        }], model, logger)

        schema.validate(config_list)

class L1FilterPruner(_StructuredFilterPruner):
    def __init__(self, model, config_list, optimizer=None):
        super().__init__(model, config_list, pruning_algorithm='l1', optimizer=optimizer)

class L2FilterPruner(_StructuredFilterPruner):
    def __init__(self, model, config_list, optimizer=None):
        super().__init__(model, config_list, pruning_algorithm='l2', optimizer=optimizer)

class FPGMPruner(_StructuredFilterPruner):
    def __init__(self, model, config_list, optimizer=None):
        super().__init__(model, config_list, pruning_algorithm='fpgm', optimizer=optimizer)

class TaylorFOWeightFilterPruner(_StructuredFilterPruner):
    def __init__(self, model, config_list, optimizer=None, statistics_batch_num=1):
        super().__init__(model, config_list, pruning_algorithm='taylorfo', optimizer=optimizer)
        self.statistics_batch_num = statistics_batch_num
        self.set_wrappers_attribute("contribution", None)
        self.iterations = 0
        self.patch_optimizer(self.calc_contributions)

    def _do_calc_mask(self, weight, bias=None, sparsity=1., wrapper=None):
        if self.iterations < self.statistics_batch_num:
            return None
        assert wrapper is not None
        if wrapper.contribution is None:
            return None

        return self.masker.calc_mask(weight, bias=bias, sparsity=sparsity, wrapper=wrapper)

    def calc_contributions(self):
        """
        Calculate the estimated importance of filters as a sum of individual contribution
        based on the first order taylor expansion.
        """
        if self.iterations >= self.statistics_batch_num:
            return
        for wrapper in self.get_modules_wrapper():
            filters = wrapper.module.weight.size(0)
            contribution = (wrapper.module.weight*wrapper.module.weight.grad).data.pow(2).view(filters, -1).sum(dim=1)
            if wrapper.contribution is None:
                wrapper.contribution = contribution
            else:
                wrapper.contribution += contribution

        self.iterations += 1
