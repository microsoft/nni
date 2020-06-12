# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from schema import And, Optional
from .constants import MASKER_DICT
from ..utils.config_validation import CompressorSchema
from ..compressor import Pruner

__all__ = ['LevelPruner', 'SlimPruner', 'L1FilterPruner', 'L2FilterPruner', 'FPGMPruner', \
    'TaylorFOWeightFilterPruner', 'ActivationAPoZRankFilterPruner', 'ActivationMeanRankFilterPruner']

logger = logging.getLogger('torch pruner')

class OneshotPruner(Pruner):
    """
    Prune model to an exact pruning level for one time.
    """

    def __init__(self, model, config_list, pruning_algorithm='level', optimizer=None, **algo_kwargs):
        """
        Parameters
        ----------
        model : torch.nn.module
            Model to be pruned
        config_list : list
            List on pruning configs
        pruning_algorithm: str
            algorithms being used to prune model
        optimizer: torch.optim.Optimizer
            Optimizer used to train model
        algo_kwargs: dict
            Additional parameters passed to pruning algorithm masker class
        """

        super().__init__(model, config_list, optimizer)
        self.set_wrappers_attribute("if_calculated", False)
        self.masker = MASKER_DICT[pruning_algorithm](model, self, **algo_kwargs)

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

    def calc_mask(self, wrapper, wrapper_idx=None):
        """
        Calculate the mask of given layer
        Parameters
        ----------
        wrapper : Module
            the module to instrument the compression operation
        wrapper_idx: int
            index of this wrapper in pruner's all wrappers
        Returns
        -------
        dict
            dictionary for storing masks, keys of the dict:
            'weight_mask':  weight mask tensor
            'bias_mask': bias mask tensor (optional)
        """
        if wrapper.if_calculated:
            return None

        sparsity = wrapper.config['sparsity']
        if not wrapper.if_calculated:
            masks = self.masker.calc_mask(sparsity=sparsity, wrapper=wrapper, wrapper_idx=wrapper_idx)

            # masker.calc_mask returns None means calc_mask is not calculated sucessfully, can try later
            if masks is not None:
                wrapper.if_calculated = True
            return masks
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

        if len(config_list) > 1:
            logger.warning('Slim pruner only supports 1 configuration')

class _StructuredFilterPruner(OneshotPruner):
    def __init__(self, model, config_list, pruning_algorithm, optimizer=None, **algo_kwargs):
        super().__init__(model, config_list, pruning_algorithm=pruning_algorithm, optimizer=optimizer, **algo_kwargs)

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
        super().__init__(model, config_list, pruning_algorithm='taylorfo', optimizer=optimizer, statistics_batch_num=statistics_batch_num)

class ActivationAPoZRankFilterPruner(_StructuredFilterPruner):
    def __init__(self, model, config_list, optimizer=None, activation='relu', statistics_batch_num=1):
        super().__init__(model, config_list, pruning_algorithm='apoz', optimizer=optimizer, \
            activation=activation, statistics_batch_num=statistics_batch_num)

class ActivationMeanRankFilterPruner(_StructuredFilterPruner):
    def __init__(self, model, config_list, optimizer=None, activation='relu', statistics_batch_num=1):
        super().__init__(model, config_list, pruning_algorithm='mean_activation', optimizer=optimizer, \
            activation=activation, statistics_batch_num=statistics_batch_num)
