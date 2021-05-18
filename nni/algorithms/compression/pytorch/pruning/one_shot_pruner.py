# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

import torch
from nni.common.graph_utils import TorchModuleGraph
from nni.compression.pytorch.compressor import Pruner
from nni.compression.pytorch.utils.config_validation import CompressorSchema
from nni.compression.pytorch.utils.shape_dependency import (ChannelDependency,
                                                            GroupDependency)
from schema import And, Optional, SchemaError
from .dependency_aware_pruner import DependencyAwarePruner

__all__ = ['LevelPruner', 'L1FilterPruner', 'L2FilterPruner', 'FPGMPruner']

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OneshotPruner(DependencyAwarePruner):
    """
    Prune model to an exact pruning level for one time.
    """

    def __init__(self, model, config_list, optimizer=None, pruning_algorithm='level', dependency_aware=False, dummy_input=None, **algo_kwargs):
        """
        Parameters
        ----------
        model : torch.nn.Module
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
        super().__init__(model, config_list, optimizer, pruning_algorithm, dependency_aware, dummy_input, **algo_kwargs)
        
        if optimizer is not None:
            self.patch_optimizer(self.update_mask)

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
        print("-> one shot pruner calc mask")
        sparsity = wrapper.config['sparsity']
        if not wrapper.if_calculated:
            masks = self.masker.calc_mask(
                sparsity=sparsity, wrapper=wrapper, wrapper_idx=wrapper_idx)

            # masker.calc_mask returns None means calc_mask is not calculated sucessfully, can try later
            if masks is not None:
                wrapper.if_calculated = True
            return masks
        else:
            return None

    def _get_threshold(self):
        """
        Calculate the global threshold to decide the weights to be pruned
        """
        pass


class LevelPruner(OneshotPruner):
    """
    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned
    config_list : list
        Supported keys:
            - sparsity : This is to specify the sparsity operations to be compressed to.
            - op_types : Operation types to prune.
    optimizer: torch.optim.Optimizer
            Optimizer used to train model
    """

    def __init__(self, model, config_list, optimizer=None):
        super().__init__(model, config_list, pruning_algorithm='level', optimizer=optimizer)


class L1FilterPruner(OneshotPruner):
    """
    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned
    config_list : list
        Supported keys:
            - sparsity : This is to specify the sparsity operations to be compressed to.
            - op_types : Only Conv2d is supported in L1FilterPruner.
    optimizer: torch.optim.Optimizer
            Optimizer used to train model
    dependency_aware: bool
        If prune the model in a dependency-aware way. If it is `True`, this pruner will
        prune the model according to the l2-norm of weights and the channel-dependency or
        group-dependency of the model. In this way, the pruner will force the conv layers
        that have dependencies to prune the same channels, so the speedup module can better
        harvest the speed benefit from the pruned model. Note that, if this flag is set True
        , the dummy_input cannot be None, because the pruner needs a dummy input to trace the
        dependency between the conv layers.
    dummy_input : torch.Tensor
        The dummy input to analyze the topology constraints. Note that, the dummy_input
        should on the same device with the model.
    """

    def __init__(self, model, config_list, optimizer=None, trainer=None, criterion=None, dependency_aware=False, dummy_input=None):
        super().__init__(model, config_list, pruning_algorithm='l1', optimizer=optimizer,
                         dependency_aware=dependency_aware, dummy_input=dummy_input)


class L2FilterPruner(OneshotPruner):
    """
    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned
    config_list : list
        Supported keys:
            - sparsity : This is to specify the sparsity operations to be compressed to.
            - op_types : Only Conv2d is supported in L2FilterPruner.
    optimizer: torch.optim.Optimizer
            Optimizer used to train model
    dependency_aware: bool
        If prune the model in a dependency-aware way. If it is `True`, this pruner will
        prune the model according to the l2-norm of weights and the channel-dependency or
        group-dependency of the model. In this way, the pruner will force the conv layers
        that have dependencies to prune the same channels, so the speedup module can better
        harvest the speed benefit from the pruned model. Note that, if this flag is set True
        , the dummy_input cannot be None, because the pruner needs a dummy input to trace the
        dependency between the conv layers.
    dummy_input : torch.Tensor
        The dummy input to analyze the topology constraints. Note that, the dummy_input
        should on the same device with the model.
    """

    def __init__(self, model, config_list, optimizer=None, trainer=None, criterion=None, dependency_aware=False, dummy_input=None):
        super().__init__(model, config_list, pruning_algorithm='l2', optimizer=optimizer,
                         dependency_aware=dependency_aware, dummy_input=dummy_input)


class FPGMPruner(OneshotPruner):
    """
    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned
    config_list : list
        Supported keys:
            - sparsity : This is to specify the sparsity operations to be compressed to.
            - op_types : Only Conv2d is supported in FPGM Pruner.
    optimizer: torch.optim.Optimizer
            Optimizer used to train model
    dependency_aware: bool
        If prune the model in a dependency-aware way. If it is `True`, this pruner will
        prune the model according to the l2-norm of weights and the channel-dependency or
        group-dependency of the model. In this way, the pruner will force the conv layers
        that have dependencies to prune the same channels, so the speedup module can better
        harvest the speed benefit from the pruned model. Note that, if this flag is set True
        , the dummy_input cannot be None, because the pruner needs a dummy input to trace the
        dependency between the conv layers.
    dummy_input : torch.Tensor
        The dummy input to analyze the topology constraints. Note that, the dummy_input
        should on the same device with the model.
    """

    def __init__(self, model, config_list, optimizer=None, dependency_aware=False, dummy_input=None):
        super().__init__(model, config_list, pruning_algorithm='fpgm',
                         dependency_aware=dependency_aware, dummy_input=dummy_input, optimizer=optimizer)





