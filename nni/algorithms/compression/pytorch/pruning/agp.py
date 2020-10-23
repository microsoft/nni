# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
An automated gradual pruning algorithm that prunes the smallest magnitude
weights to achieve a preset level of network sparsity.
Michael Zhu and Suyog Gupta, "To prune, or not to prune: exploring the
efficacy of pruning for model compression", 2017 NIPS Workshop on Machine
Learning of Phones and other Consumer Devices.
"""

import logging
import torch
from schema import And, Optional
from .constants import MASKER_DICT
from nni.compression.pytorch.utils.config_validation import CompressorSchema
from nni.compression.pytorch.compressor import Pruner

__all__ = ['AGPPruner']

logger = logging.getLogger('torch pruner')

class AGPPruner(Pruner):
    """
    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned.
    config_list : listlist
        Supported keys:
            - initial_sparsity: This is to specify the sparsity when compressor starts to compress.
            - final_sparsity: This is to specify the sparsity when compressor finishes to compress.
            - start_epoch: This is to specify the epoch number when compressor starts to compress, default start from epoch 0.
            - end_epoch: This is to specify the epoch number when compressor finishes to compress.
            - frequency: This is to specify every *frequency* number epochs compressor compress once, default frequency=1.
    optimizer: torch.optim.Optimizer
        Optimizer used to train model.
    pruning_algorithm: str
        Algorithms being used to prune model,
        choose from `['level', 'slim', 'l1', 'l2', 'fpgm', 'taylorfo', 'apoz', 'mean_activation']`, by default `level`
    """

    def __init__(self, model, config_list, optimizer, pruning_algorithm='level'):
        super().__init__(model, config_list, optimizer)
        assert isinstance(optimizer, torch.optim.Optimizer), "AGP pruner is an iterative pruner, please pass optimizer of the model to it"
        self.masker = MASKER_DICT[pruning_algorithm](model, self)

        self.now_epoch = 0
        self.set_wrappers_attribute("if_calculated", False)

    def validate_config(self, model, config_list):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Model to be pruned
        config_list : list
            List on pruning configs
        """
        schema = CompressorSchema([{
            'initial_sparsity': And(float, lambda n: 0 <= n <= 1),
            'final_sparsity': And(float, lambda n: 0 <= n <= 1),
            'start_epoch': And(int, lambda n: n >= 0),
            'end_epoch': And(int, lambda n: n >= 0),
            'frequency': And(int, lambda n: n > 0),
            Optional('op_types'): [str],
            Optional('op_names'): [str]
        }], model, logger)

        schema.validate(config_list)

    def calc_mask(self, wrapper, wrapper_idx=None):
        """
        Calculate the mask of given layer.
        Scale factors with the smallest absolute value in the BN layer are masked.
        Parameters
        ----------
        wrapper : Module
            the layer to instrument the compression operation
        wrapper_idx: int
            index of this wrapper in pruner's all wrappers
        Returns
        -------
        dict | None
            Dictionary for storing masks, keys of the dict:
            'weight_mask':  weight mask tensor
            'bias_mask': bias mask tensor (optional)
        """

        config = wrapper.config

        start_epoch = config.get('start_epoch', 0)
        freq = config.get('frequency', 1)

        if wrapper.if_calculated:
            return None
        if not (self.now_epoch >= start_epoch and (self.now_epoch - start_epoch) % freq == 0):
            return None

        target_sparsity = self.compute_target_sparsity(config)
        new_mask = self.masker.calc_mask(sparsity=target_sparsity, wrapper=wrapper, wrapper_idx=wrapper_idx)
        if new_mask is not None:
            wrapper.if_calculated = True

        return new_mask

    def compute_target_sparsity(self, config):
        """
        Calculate the sparsity for pruning
        Parameters
        ----------
        config : dict
            Layer's pruning config
        Returns
        -------
        float
            Target sparsity to be pruned
        """

        end_epoch = config.get('end_epoch', 1)
        start_epoch = config.get('start_epoch', 0)
        freq = config.get('frequency', 1)
        final_sparsity = config.get('final_sparsity', 0)
        initial_sparsity = config.get('initial_sparsity', 0)
        if end_epoch <= start_epoch or initial_sparsity >= final_sparsity:
            logger.warning('your end epoch <= start epoch or initial_sparsity >= final_sparsity')
            return final_sparsity

        if end_epoch <= self.now_epoch:
            return final_sparsity

        span = ((end_epoch - start_epoch - 1) // freq) * freq
        assert span > 0
        target_sparsity = (final_sparsity +
                           (initial_sparsity - final_sparsity) *
                           (1.0 - ((self.now_epoch - start_epoch) / span)) ** 3)
        return target_sparsity

    def update_epoch(self, epoch):
        """
        Update epoch
        Parameters
        ----------
        epoch : int
            current training epoch
        """

        if epoch > 0:
            self.now_epoch = epoch
            for wrapper in self.get_modules_wrapper():
                wrapper.if_calculated = False
