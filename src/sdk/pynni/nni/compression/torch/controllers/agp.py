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

__all__ = ['AGP_Pruner']

logger = logging.getLogger('torch pruner')

class AGP_Pruner(Pruner):
    """
    An automated gradual pruning algorithm that prunes the smallest magnitude
    weights to achieve a preset level of network sparsity.
    Michael Zhu and Suyog Gupta, "To prune, or not to prune: exploring the
    efficacy of pruning for model compression", 2017 NIPS Workshop on Machine
    Learning of Phones and other Consumer Devices,
    https://arxiv.org/pdf/1710.01878.pdf
    """

    def __init__(self, model, config_list, optimizer, pruning_algorithm='level'):
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
        assert isinstance(optimizer, torch.optim.Optimizer), "AGP pruner is an iterative pruner, please pass optimizer of the model to it"
        self.masker = masker_dict[pruning_algorithm](model, self)

        self.now_epoch = 0
        self.set_wrappers_attribute("if_calculated", False)

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
            'initial_sparsity': And(float, lambda n: 0 <= n <= 1),
            'final_sparsity': And(float, lambda n: 0 <= n <= 1),
            'start_epoch': And(int, lambda n: n >= 0),
            'end_epoch': And(int, lambda n: n >= 0),
            'frequency': And(int, lambda n: n > 0),
            Optional('op_types'): [str],
            Optional('op_names'): [str]
        }], model, logger)

        schema.validate(config_list)

    def calc_mask(self, wrapper, **kwargs):
        """
        Calculate the mask of given layer.
        Scale factors with the smallest absolute value in the BN layer are masked.
        Parameters
        ----------
        wrapper : Module
            the layer to instrument the compression operation

        Returns
        -------
        dict
            dictionary for storing masks
        """

        config = wrapper.config
        weight = wrapper.module.weight.data
        start_epoch = config.get('start_epoch', 0)
        freq = config.get('frequency', 1)

        if wrapper.if_calculated:
            return None
        if not (self.now_epoch >= start_epoch and (self.now_epoch - start_epoch) % freq == 0):
            return None

        mask = {'weight_mask': wrapper.weight_mask}
        target_sparsity = self.compute_target_sparsity(config)
        k = int(weight.numel() * target_sparsity)
        if k == 0 or target_sparsity >= 1 or target_sparsity <= 0:
            return mask
        # if we want to generate new mask, we should update weigth first
        #w_abs = weight.abs() * mask['weight_mask']
        #threshold = torch.topk(w_abs.view(-1), k, largest=False)[0].max()
        #new_mask = {'weight_mask': torch.gt(w_abs, threshold).type_as(weight)}
        new_mask = self.masker.calc_mask(weight, sparsity=target_sparsity)
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
