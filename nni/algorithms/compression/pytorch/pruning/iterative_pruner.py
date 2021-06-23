# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import copy
import torch
from schema import And, Optional
from nni.compression.pytorch.utils.config_validation import CompressorSchema
from .constants import MASKER_DICT
from .dependency_aware_pruner import DependencyAwarePruner

__all__ = ['AGPPruner', 'ADMMPruner', 'SlimPruner', 'TaylorFOWeightFilterPruner', 'ActivationAPoZRankFilterPruner',
           'ActivationMeanRankFilterPruner']

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class IterativePruner(DependencyAwarePruner):
    """
    Prune model during the training process.
    """

    def __init__(self, model, config_list, optimizer=None, pruning_algorithm='slim', trainer=None, criterion=None,
                 num_iterations=20, epochs_per_iteration=5, dependency_aware=False, dummy_input=None, **algo_kwargs):
        """
        Parameters
        ----------
        model: torch.nn.Module
            Model to be pruned
        config_list: list
            List on pruning configs
        optimizer: torch.optim.Optimizer
            Optimizer used to train model
        pruning_algorithm: str
            algorithms being used to prune model
        trainer: function
            Function used to train the model.
            Users should write this function as a normal function to train the Pytorch model
            and include `model, optimizer, criterion, epoch` as function arguments.
        criterion: function
            Function used to calculate the loss between the target and the output.
            For example, you can use ``torch.nn.CrossEntropyLoss()`` as input.
        num_iterations: int
            Total number of iterations in pruning process. We will calculate mask at the end of an iteration.
        epochs_per_iteration: Union[int, list]
            The number of training epochs for each iteration. `int` represents the same value for each iteration.
            `list` represents the specific value for each iteration.
        dependency_aware: bool
            If prune the model in a dependency-aware way.
        dummy_input: torch.Tensor
            The dummy input to analyze the topology constraints. Note that,
            the dummy_input should on the same device with the model.
        algo_kwargs: dict
            Additional parameters passed to pruning algorithm masker class
        """
        super().__init__(model, config_list, optimizer, pruning_algorithm, dependency_aware, dummy_input, **algo_kwargs)

        if isinstance(epochs_per_iteration, list):
            assert len(epochs_per_iteration) == num_iterations, 'num_iterations should equal to the length of epochs_per_iteration'
            self.epochs_per_iteration = epochs_per_iteration
        else:
            assert num_iterations > 0, 'num_iterations should >= 1'
            self.epochs_per_iteration = [epochs_per_iteration] * num_iterations

        self._validate_iteration_params()

        self._trainer = trainer
        self._criterion = criterion

    def _fresh_calculated(self):
        for wrapper in self.get_modules_wrapper():
            wrapper.if_calculated = False

    def _validate_iteration_params(self):
        assert all(num >= 0 for num in self.epochs_per_iteration), 'all epoch number need >= 0'

    def compress(self):
        training = self.bound_model.training
        self.bound_model.train()
        for _, epochs_num in enumerate(self.epochs_per_iteration):
            self._fresh_calculated()
            for epoch in range(epochs_num):
                self._trainer(self.bound_model, optimizer=self.optimizer, criterion=self._criterion, epoch=epoch)
            # NOTE: workaround for statistics_batch_num bigger than max batch number in one epoch, need refactor
            if hasattr(self.masker, 'statistics_batch_num') and hasattr(self, 'iterations'):
                if self.iterations < self.masker.statistics_batch_num:
                    self.iterations = self.masker.statistics_batch_num
            self.update_mask()
        self.bound_model.train(training)

        return self.bound_model


class AGPPruner(IterativePruner):
    """
    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned.
    config_list : listlist
        Supported keys:
            - sparsity : This is to specify the sparsity operations to be compressed to.
            - op_types : See supported type in your specific pruning algorithm.
    optimizer: torch.optim.Optimizer
        Optimizer used to train model.
    trainer: function
        Function to train the model
    criterion: function
        Function used to calculate the loss between the target and the output.
        For example, you can use ``torch.nn.CrossEntropyLoss()`` as input.
    num_iterations: int
        Total number of iterations in pruning process. We will calculate mask at the end of an iteration.
    epochs_per_iteration: int
        The number of training epochs for each iteration.
    pruning_algorithm: str
        Algorithms being used to prune model,
        choose from `['level', 'slim', 'l1', 'l2', 'fpgm', 'taylorfo', 'apoz', 'mean_activation']`, by default `level`
    """

    def __init__(self, model, config_list, optimizer, trainer, criterion, num_iterations=10, epochs_per_iteration=1, pruning_algorithm='level'):
        super().__init__(model, config_list, optimizer=optimizer, trainer=trainer, criterion=criterion,
                         num_iterations=num_iterations, epochs_per_iteration=epochs_per_iteration)
        assert isinstance(optimizer, torch.optim.Optimizer), "AGP pruner is an iterative pruner, please pass optimizer of the model to it"
        self.masker = MASKER_DICT[pruning_algorithm](model, self)
        self.now_epoch = 0
        self.freq = epochs_per_iteration
        self.end_epoch = epochs_per_iteration * num_iterations
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
            'sparsity': And(float, lambda n: 0 <= n <= 1),
            Optional('op_types'): [str],
            Optional('op_names'): [str]
        }], model, logger)

        schema.validate(config_list)

    def _supported_dependency_aware(self):
        return False

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

        if wrapper.if_calculated:
            return None

        if not self.now_epoch % self.freq == 0:
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

        initial_sparsity = 0
        self.target_sparsity = final_sparsity = config.get('sparsity', 0)

        if initial_sparsity >= final_sparsity:
            logger.warning('your initial_sparsity >= final_sparsity')
            return final_sparsity

        if self.end_epoch == 1 or self.end_epoch <= self.now_epoch:
            return final_sparsity

        span = ((self.end_epoch - 1) // self.freq) * self.freq
        assert span > 0
        self.target_sparsity = (final_sparsity + (initial_sparsity - final_sparsity) * (1.0 - (self.now_epoch / span)) ** 3)
        return self.target_sparsity

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

    # TODO: need refactor
    def compress(self):
        training = self.bound_model.training
        self.bound_model.train()

        for epoch in range(self.end_epoch):
            self.update_epoch(epoch)
            self._trainer(self.bound_model, optimizer=self.optimizer, criterion=self._criterion, epoch=epoch)
            self.update_mask()
            logger.info(f'sparsity is {self.target_sparsity:.2f} at epoch {epoch}')
            self.get_pruned_weights()

        self.bound_model.train(training)

        return self.bound_model


class ADMMPruner(IterativePruner):
    """
    A Pytorch implementation of ADMM Pruner algorithm.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned.
    config_list : list
        List on pruning configs.
    trainer : function
        Function used for the first subproblem.
        Users should write this function as a normal function to train the Pytorch model
        and include `model, optimizer, criterion, epoch` as function arguments.
    criterion: function
        Function used to calculate the loss between the target and the output. By default, we use CrossEntropyLoss in ADMMPruner.
        For example, you can use ``torch.nn.CrossEntropyLoss()`` as input.
    num_iterations: int
        Total number of iterations in pruning process. We will calculate mask after we finish all iterations in ADMMPruner.
    epochs_per_iteration: int
        Training epochs of the first subproblem.
    row : float
        Penalty parameters for ADMM training.
    base_algo : str
        Base pruning algorithm. `level`, `l1`, `l2` or `fpgm`, by default `l1`. Given the sparsity distribution among
        the ops, the assigned `base_algo` is used to decide which filters/channels/weights to prune.
    """

    def __init__(self, model, config_list, trainer, criterion=torch.nn.CrossEntropyLoss(),
                 num_iterations=30, epochs_per_iteration=5, row=1e-4, base_algo='l1'):
        self._base_algo = base_algo

        super().__init__(model, config_list)

        self._trainer = trainer
        self.optimizer = torch.optim.Adam(
            self.bound_model.parameters(), lr=1e-3, weight_decay=5e-5)
        self._criterion = criterion
        self._num_iterations = num_iterations
        self._training_epochs = epochs_per_iteration
        self._row = row

        self.set_wrappers_attribute("if_calculated", False)
        self.masker = MASKER_DICT[self._base_algo](self.bound_model, self)

        self.patch_optimizer_before(self._callback)

    def validate_config(self, model, config_list):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Model to be pruned
        config_list : list
            List on pruning configs
        """

        if self._base_algo == 'level':
            schema = CompressorSchema([{
                'sparsity': And(float, lambda n: 0 < n < 1),
                Optional('op_types'): [str],
                Optional('op_names'): [str],
            }], model, logger)
        elif self._base_algo in ['l1', 'l2', 'fpgm']:
            schema = CompressorSchema([{
                'sparsity': And(float, lambda n: 0 < n < 1),
                'op_types': ['Conv2d'],
                Optional('op_names'): [str]
            }], model, logger)

        schema.validate(config_list)

    def _supported_dependency_aware(self):
        return False

    def _projection(self, weight, sparsity, wrapper):
        '''
        Return the Euclidean projection of the weight matrix according to the pruning mode.

        Parameters
        ----------
        weight : tensor
            original matrix
        sparsity : float
            the ratio of parameters which need to be set to zero
        wrapper: PrunerModuleWrapper
            layer wrapper of this layer

        Returns
        -------
        tensor
            the projected matrix
        '''
        wrapper_copy = copy.deepcopy(wrapper)
        wrapper_copy.module.weight.data = weight
        return weight.data.mul(self.masker.calc_mask(sparsity, wrapper_copy)['weight_mask'])

    def _callback(self):
        # callback function to do additonal optimization, refer to the deriatives of Formula (7)
        for i, wrapper in enumerate(self.get_modules_wrapper()):
            wrapper.module.weight.data -= self._row * \
                (wrapper.module.weight.data - self.Z[i] + self.U[i])

    def compress(self):
        """
        Compress the model with ADMM.

        Returns
        -------
        torch.nn.Module
            model with specified modules compressed.
        """
        logger.info('Starting ADMM Compression...')

        # initiaze Z, U
        # Z_i^0 = W_i^0
        # U_i^0 = 0
        self.Z = []
        self.U = []
        for wrapper in self.get_modules_wrapper():
            z = wrapper.module.weight.data
            self.Z.append(z)
            self.U.append(torch.zeros_like(z))

        # Loss = cross_entropy +  l2 regulization + \Sum_{i=1}^N \row_i ||W_i - Z_i^k + U_i^k||^2
        # optimization iteration
        for k in range(self._num_iterations):
            logger.info('ADMM iteration : %d', k)

            # step 1: optimize W with AdamOptimizer
            for epoch in range(self._training_epochs):
                self._trainer(self.bound_model, optimizer=self.optimizer, criterion=self._criterion, epoch=epoch)

            # step 2: update Z, U
            # Z_i^{k+1} = projection(W_i^{k+1} + U_i^k)
            # U_i^{k+1} = U^k + W_i^{k+1} - Z_i^{k+1}
            for i, wrapper in enumerate(self.get_modules_wrapper()):
                z = wrapper.module.weight.data + self.U[i]
                self.Z[i] = self._projection(z, wrapper.config['sparsity'], wrapper)
                self.U[i] = self.U[i] + wrapper.module.weight.data - self.Z[i]

        # apply prune
        self.update_mask()

        logger.info('Compression finished.')

        return self.bound_model


class SlimPruner(IterativePruner):
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
        and include `model, optimizer, criterion, epoch` as function arguments.
    criterion : function
        Function used to calculate the loss between the target and the output.
        For example, you can use ``torch.nn.CrossEntropyLoss()`` as input.
    sparsifying_training_epochs: int
        The number of channel sparsity regularization training epochs before pruning.
    scale : float
        Penalty parameters for sparsification.
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

    def __init__(self, model, config_list, optimizer, trainer, criterion, sparsifying_training_epochs=10, scale=0.0001,
                 dependency_aware=False, dummy_input=None):
        super().__init__(model, config_list, optimizer=optimizer, pruning_algorithm='slim', trainer=trainer, criterion=criterion,
                         num_iterations=1, epochs_per_iteration=sparsifying_training_epochs, dependency_aware=dependency_aware,
                         dummy_input=dummy_input)
        self.scale = scale
        self.patch_optimizer_before(self._callback)

    def validate_config(self, model, config_list):
        schema = CompressorSchema([{
            'sparsity': And(float, lambda n: 0 < n < 1),
            'op_types': ['BatchNorm2d'],
            Optional('op_names'): [str]
        }], model, logger)

        schema.validate(config_list)

        if len(config_list) > 1:
            logger.warning('Slim pruner only supports 1 configuration')

    def _supported_dependency_aware(self):
        return True

    def _callback(self):
        for _, wrapper in enumerate(self.get_modules_wrapper()):
            wrapper.module.weight.grad.data.add_(self.scale * torch.sign(wrapper.module.weight.data))


class TaylorFOWeightFilterPruner(IterativePruner):
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
    trainer : function
        Function used to sparsify BatchNorm2d scaling factors.
        Users should write this function as a normal function to train the Pytorch model
        and include `model, optimizer, criterion, epoch` as function arguments.
    criterion : function
        Function used to calculate the loss between the target and the output.
        For example, you can use ``torch.nn.CrossEntropyLoss()`` as input.
    sparsifying_training_batches: int
        The number of batches to collect the contributions. Note that the number need to be less than the maximum batch number in one epoch.
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

    def __init__(self, model, config_list, optimizer, trainer, criterion, sparsifying_training_batches=1,
                 dependency_aware=False, dummy_input=None):
        super().__init__(model, config_list, optimizer=optimizer, pruning_algorithm='taylorfo', trainer=trainer,
                         criterion=criterion, statistics_batch_num=sparsifying_training_batches, num_iterations=1,
                         epochs_per_iteration=1, dependency_aware=dependency_aware,
                         dummy_input=dummy_input)

    def _supported_dependency_aware(self):
        return True


class ActivationAPoZRankFilterPruner(IterativePruner):
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
        and include `model, optimizer, criterion, epoch` as function arguments.
    criterion : function
        Function used to calculate the loss between the target and the output.
        For example, you can use ``torch.nn.CrossEntropyLoss()`` as input.
    activation: str
        The activation type.
    sparsifying_training_batches: int
        The number of batches to collect the contributions. Note that the number need to be less than the maximum batch number in one epoch.
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

    def __init__(self, model, config_list, optimizer, trainer, criterion, activation='relu',
                 sparsifying_training_batches=1, dependency_aware=False, dummy_input=None):
        super().__init__(model, config_list, pruning_algorithm='apoz', optimizer=optimizer, trainer=trainer,
                         criterion=criterion, dependency_aware=dependency_aware, dummy_input=dummy_input,
                         activation=activation, statistics_batch_num=sparsifying_training_batches, num_iterations=1,
                         epochs_per_iteration=1)
        self.patch_optimizer(self.update_mask)

    def _supported_dependency_aware(self):
        return True


class ActivationMeanRankFilterPruner(IterativePruner):
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
            and include `model, optimizer, criterion, epoch` as function arguments.
    criterion : function
        Function used to calculate the loss between the target and the output.
        For example, you can use ``torch.nn.CrossEntropyLoss()`` as input.
    activation: str
        The activation type.
    sparsifying_training_batches: int
        The number of batches to collect the contributions. Note that the number need to be less than the maximum batch number in one epoch.
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

    def __init__(self, model, config_list, optimizer, trainer, criterion, activation='relu',
                 sparsifying_training_batches=1, dependency_aware=False, dummy_input=None):
        super().__init__(model, config_list, pruning_algorithm='mean_activation', optimizer=optimizer, trainer=trainer,
                         criterion=criterion, dependency_aware=dependency_aware, dummy_input=dummy_input,
                         activation=activation, statistics_batch_num=sparsifying_training_batches, num_iterations=1,
                         epochs_per_iteration=1)
        self.patch_optimizer(self.update_mask)

    def _supported_dependency_aware(self):
        return True
