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

__all__ = ['LevelPruner', 'SlimPruner', 'L1FilterPruner', 'L2FilterPruner', 'FPGMPruner']

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
            wrapper.if_calculated = True
            return self.masker.calc_mask(weight, bias=bias, sparsity=sparsity)
        else:
            return None

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

class _WeightRankPruner(OneshotPruner):
    def __init__(self, model, config_list, pruning_algorithm, optimizer=None):
        super().__init__(model, config_list, pruning_algorithm=pruning_algorithm, optimizer=optimizer)

    def validate_config(self, model, config_list):
        schema = CompressorSchema([{
            'sparsity': And(float, lambda n: 0 < n < 1),
            'op_types': ['Conv2d'],
            Optional('op_names'): [str]
        }], model, logger)

        schema.validate(config_list)

class L1FilterPruner(_WeightRankPruner):
    def __init__(self, model, config_list, optimizer=None):
        super().__init__(model, config_list, pruning_algorithm='l1', optimizer=optimizer)

class L2FilterPruner(_WeightRankPruner):
    def __init__(self, model, config_list, optimizer=None):
        super().__init__(model, config_list, pruning_algorithm='l2', optimizer=optimizer)

class FPGMPruner(_WeightRankPruner):
    def __init__(self, model, config_list, optimizer=None):
        super().__init__(model, config_list, pruning_algorithm='fpgm', optimizer=optimizer)
