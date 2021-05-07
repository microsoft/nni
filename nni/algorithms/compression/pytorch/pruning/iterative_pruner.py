# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import torch
from schema import And, Optional, SchemaError
from nni.compression.pytorch.utils.config_validation import CompressorSchema
from .one_shot_pruner import OneshotPruner, _StructuredFilterPruner

__all__ = ['SlimPruner', 'TaylorFOWeightFilterPruner', 'ActivationAPoZRankFilterPruner', 'ActivationMeanRankFilterPruner']

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SlimPruner(OneshotPruner):
    """
    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned
    config_list : list
        Supported keys:
            - sparsity : This is to specify the sparsity operations to be compressed to.
            - op_types : Only BatchNorm2d is supported in Slim Pruner.
    optimizer : torch.optim.Optimizer
            Optimizer used to train model
    trainer : function
        Function used to sparsify BatchNorm2d scaling factors.
        Users should write this function as a normal function to train the Pytorch model
        and include `model, optimizer, criterion, epoch, callback` as function arguments.
    criterion : function
        Function used to calculate the loss between the target and the output.
    training_epochs : int
        Totoal number of epochs for sparsification.
    scale : float 
        Penalty parameters for sparsification.
    """

    def __init__(self, model, config_list, optimizer, trainer, criterion, training_epochs=2, scale=0.0001):
        super().__init__(model, config_list, pruning_algorithm='slim')
        self.training_aware = True
        self.training_epochs = training_epochs
        self.scale = scale
        self.optimizer = optimizer
        self._trainer = trainer
        self._criterion = criterion

    def validate_config(self, model, config_list):
        schema = CompressorSchema([{
            'sparsity': And(float, lambda n: 0 < n < 1),
            'op_types': ['BatchNorm2d'],
            Optional('op_names'): [str]
        }], model, logger)

        schema.validate(config_list)

        if len(config_list) > 1:
            logger.warning('Slim pruner only supports 1 configuration')

    def _get_threshold(self):
        weight_list = []
        for (layer, _) in self.get_modules_to_compress():
            weight_list.append(layer.module.weight.data.abs().clone())
        all_bn_weights = torch.cat(weight_list)
        k = int(all_bn_weights.shape[0] * self.config_list[0]['sparsity'])
        self.masker.global_threshold = torch.topk(
            all_bn_weights.view(-1), k, largest=False)[0].max()

    def _callback(self):
        for i, wrapper in enumerate(self.get_modules_wrapper()):
            wrapper.module.weight.grad.data.add_(self.scale * torch.sign(wrapper.module.weight.data))


class TaylorFOWeightFilterPruner(_StructuredFilterPruner):
    """
    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned
    config_list : list
        Supported keys:
            - sparsity : How much percentage of convolutional filters are to be pruned.
            - op_types : Currently only Conv2d is supported in TaylorFOWeightFilterPruner.
    optimizer: torch.optim.Optimizer
            Optimizer used to train model
    training_epochs: int
        The number of epochs to calculate the contributions.
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

    def __init__(self, model, config_list, optimizer, trainer, criterion, training_epochs=1,
                 dependency_aware=False, dummy_input=None):
        super().__init__(model, config_list, pruning_algorithm='taylorfo',
                         dependency_aware=dependency_aware, dummy_input=dummy_input,
                         optimizer=optimizer, trainer=trainer, criterion=criterion)
        self.training_aware = True
        self.training_epochs = training_epochs


class ActivationAPoZRankFilterPruner(_StructuredFilterPruner):
    """
    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned
    config_list : list
        Supported keys:
            - sparsity : How much percentage of convolutional filters are to be pruned.
            - op_types : Only Conv2d is supported in ActivationAPoZRankFilterPruner.
    optimizer: torch.optim.Optimizer
            Optimizer used to train model
    trainer: function
        Function used to train the model.
        Users should write this function as a normal function to train the Pytorch model
        and include `model, optimizer, criterion, epoch, callback` as function arguments.
    training_epochs: int
        The number of epochs to statistic the activation.
    activation: str
        The activation type.
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

    def __init__(self, model, config_list, optimizer=None, trainer=None, criterion=None, activation='relu',
                 training_epochs=1, dependency_aware=False, dummy_input=None):
        super().__init__(model, config_list, pruning_algorithm='apoz', optimizer=optimizer, trainer=trainer, criterion=criterion,
                         dependency_aware=dependency_aware, dummy_input=dummy_input,
                         activation=activation, training_epochs=training_epochs)
        self.training_aware = True
        self.training_epochs = training_epochs


class ActivationMeanRankFilterPruner(_StructuredFilterPruner):
    """
    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned
    config_list : list
        Supported keys:
            - sparsity : How much percentage of convolutional filters are to be pruned.
            - op_types : Only Conv2d is supported in ActivationMeanRankFilterPruner.
    optimizer: torch.optim.Optimizer
            Optimizer used to train model.
    trainer: function
            Function used to train the model.
            Users should write this function as a normal function to train the Pytorch model
            and include `model, optimizer, criterion, epoch, callback` as function arguments.
    activation: str
        The activation type.
    training_epochs: int
        The number of batches to statistic the activation.
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

    def __init__(self, model, config_list, optimizer=None, trainer=None, criterion=None, activation='relu',
                 training_epochs=1, dependency_aware=False, dummy_input=None):
        super().__init__(model, config_list, pruning_algorithm='mean_activation', optimizer=optimizer, trainer=trainer, criterion=criterion,
                         dependency_aware=dependency_aware, dummy_input=dummy_input,
                         activation=activation, training_epochs=training_epochs)
        self.training_aware = True
        self.training_epochs = training_epochs
