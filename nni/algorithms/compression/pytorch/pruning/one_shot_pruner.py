# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from schema import And, Optional

from nni.compression.pytorch.utils.config_validation import CompressorSchema
from .dependency_aware_pruner import DependencyAwarePruner

__all__ = ['LevelPruner', 'L1FilterPruner', 'L2FilterPruner', 'FPGMPruner']

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OneshotPruner(DependencyAwarePruner):
    """
    Prune model to an exact pruning level for one time.
    """

    def __init__(self, model, config_list, pruning_algorithm='level', dependency_aware=False, dummy_input=None,
                 **algo_kwargs):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Model to be pruned
        config_list : list
            List on pruning configs
        pruning_algorithm: str
            algorithms being used to prune model
        dependency_aware: bool
            If prune the model in a dependency-aware way.
        dummy_input : torch.Tensor
            The dummy input to analyze the topology constraints. Note that,
            the dummy_input should on the same device with the model.
        algo_kwargs: dict
            Additional parameters passed to pruning algorithm masker class
        """
        super().__init__(model, config_list, None, pruning_algorithm, dependency_aware, dummy_input, **algo_kwargs)

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
    """

    def __init__(self, model, config_list):
        super().__init__(model, config_list, pruning_algorithm='level')

    def _supported_dependency_aware(self):
        return False


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

    def __init__(self, model, config_list, dependency_aware=False, dummy_input=None):
        super().__init__(model, config_list, pruning_algorithm='l1', dependency_aware=dependency_aware,
                         dummy_input=dummy_input)

    def _supported_dependency_aware(self):
        return True


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

    def __init__(self, model, config_list, dependency_aware=False, dummy_input=None):
        super().__init__(model, config_list, pruning_algorithm='l2', dependency_aware=dependency_aware,
                         dummy_input=dummy_input)

    def _supported_dependency_aware(self):
        return True


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

    def __init__(self, model, config_list, dependency_aware=False, dummy_input=None):
        super().__init__(model, config_list, pruning_algorithm='fpgm', dependency_aware=dependency_aware,
                         dummy_input=dummy_input)

    def _supported_dependency_aware(self):
        return True
