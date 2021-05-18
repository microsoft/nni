# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import torch
from schema import And, Optional, SchemaError
from nni.compression.pytorch.utils.config_validation import CompressorSchema
from nni.compression.pytorch.compressor import Pruner
from .constants import MASKER_DICT, MAX_EPOCHS
from .structured_pruner import StructuredFilterPruner

__all__ = ['AGPPruner', 'ADMMPruner', 'SlimPruner', 'TaylorFOWeightFilterPruner', 'ActivationAPoZRankFilterPruner', 'ActivationMeanRankFilterPruner', ]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class IterativePruner(Pruner):
    """
    Prune model during the training process. 
    """

    def __init__(self, model, config_list, pruning_algorithm='level', optimizer=None, trainer=None, criterion=None, training_eopchs=1, **algo_kwargs):
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
        trainer: function
            Function used to train the model.
            Users should write this function as a normal function to train the Pytorch model
            and include `model, optimizer, criterion, epoch, callback` as function arguments.
        criterion: function
            Function used to calculate the loss between the target and the output.
        training_epochs : int
            Totoal number of epochs for training.
        algo_kwargs: dict
            Additional parameters passed to pruning algorithm masker class
        """
        print("IterativePruner init Start")
        super().__init__(model, config_list, optimizer)
        self.set_wrappers_attribute("if_calculated", False)
        self._trainer = trainer
        self._criterion = criterion
        self.masker = MASKER_DICT[pruning_algorithm](
            model, self, **algo_kwargs)
        print("IterativePruner init Done")

    def compress(self):
        training = self.bound_model.training
        self.bound_model.train()
        for epoch in range(MAX_EPOCHS):
            self._trainer(self.bound_model, optimizer=self.optimizer,
            criterion=self._criterion, epoch=epoch, callback=self._callback)
            if epoch >= self.training_epochs:
                break
        self.bound_model.train(training)
        self._get_threshold()

        self.update_mask()
        return self.bound_model


    def _callback(self):
        """
        Callback function to do additonal optimization
        """
        pass


class AGPPruner(IterativePruner):
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
    trainer: function
        Function to train the model
    pruning_algorithm: str
        Algorithms being used to prune model,
        choose from `['level', 'slim', 'l1', 'l2', 'fpgm', 'taylorfo', 'apoz', 'mean_activation']`, by default `level`
    """

    def __init__(self, model, config_list, optimizer, trainer, criterion, training_epochs=1, pruning_algorithm='level'):
        super().__init__(model, config_list, optimizer=optimizer, trainer=trainer, criterion=criterion)
        assert isinstance(optimizer, torch.optim.Optimizer), "AGP pruner is an iterative pruner, please pass optimizer of the model to it"
        self.masker = MASKER_DICT[pruning_algorithm](model, self)
        self.training_epochs = training_epochs
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
            'sparsity': And(float, lambda n: 0 <= n <= 1),
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
        freq = config.get('frequency', 1)

        if wrapper.if_calculated:
            return None

        if not self.now_epoch % freq == 0:
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
        end_epoch = self.training_epochs
        freq = config.get('frequency', 1)
        self.target_sparsity = final_sparsity = config.get('sparsity', 0)
        
        if initial_sparsity >= final_sparsity:
            logger.warning('your initial_sparsity >= final_sparsity')
            return final_sparsity

        if end_epoch == 1 or end_epoch <= self.now_epoch:
            return final_sparsity
            
        span = ((end_epoch - 1) // freq) * freq
        assert span > 0
        self.target_sparsity = (final_sparsity +
                           (initial_sparsity - final_sparsity) *
                           (1.0 - ((self.now_epoch) / span)) ** 3)
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

    def compress(self):
        for epoch in range(self.training_epochs):
            self.update_epoch(epoch)
            self._trainer(self.bound_model, optimizer=self.optimizer, criterion=self._criterion, epoch=epoch)
            logger.info(f'sparsity is {self.target_sparsity:.2f} at epoch {epoch}')
            self.update_mask()
            self.get_pruned_weights()
            
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
        and include `model, optimizer, criterion, epoch, callback` as function arguments.
        Here `callback` acts as an L2 regulizer as presented in the formula (7) of the original paper.
        The logic of `callback` is implemented inside the Pruner,
        users are just required to insert `callback()` between `loss.backward()` and `optimizer.step()`.
        Example::

            def trainer(model, criterion, optimizer, epoch, callback):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                train_loader = ...
                model.train()
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    # callback should be inserted between loss.backward() and optimizer.step()
                    if callback:
                        callback()
                    optimizer.step()
    num_iterations : int
        Total number of iterations.
    training_epochs : int
        Training epochs of the first subproblem.
    row : float
        Penalty parameters for ADMM training.
    base_algo : str
        Base pruning algorithm. `level`, `l1`, `l2` or `fpgm`, by default `l1`. Given the sparsity distribution among the ops,
        the assigned `base_algo` is used to decide which filters/channels/weights to prune.

    """

    def __init__(self, model, config_list, optimizer, trainer, criterion, num_iterations=30, training_epochs=5, row=1e-4, base_algo='l1'):
        self._base_algo = base_algo

        super().__init__(model, config_list)

        self._trainer = trainer
        self._optimizer = optimizer
        self._criterion = criterion
        self._num_iterations = num_iterations
        self._training_epochs = training_epochs
        self._row = row

        self.set_wrappers_attribute("if_calculated", False)
        self.masker = MASKER_DICT[self._base_algo](self.bound_model, self)

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

    def compress(self):
        """
        Compress the model with ADMM.

        Returns
        -------
        torch.nn.Module
            model with specified modules compressed.
        """
        _logger.info('Starting ADMM Compression...')

        # initiaze Z, U
        # Z_i^0 = W_i^0
        # U_i^0 = 0
        Z = []
        U = []
        for wrapper in self.get_modules_wrapper():
            z = wrapper.module.weight.data
            Z.append(z)
            U.append(torch.zeros_like(z))

        optimizer = torch.optim.Adam(
            self.bound_model.parameters(), lr=1e-3, weight_decay=5e-5)

        # Loss = cross_entropy +  l2 regulization + \Sum_{i=1}^N \row_i ||W_i - Z_i^k + U_i^k||^2
        # criterion = torch.nn.CrossEntropyLoss()

        # callback function to do additonal optimization, refer to the deriatives of Formula (7)
        def callback():
            for i, wrapper in enumerate(self.get_modules_wrapper()):
                wrapper.module.weight.data -= self._row * \
                    (wrapper.module.weight.data - Z[i] + U[i])

        # optimization iteration
        for k in range(self._num_iterations):
            logger.info('ADMM iteration : %d', k)

            # step 1: optimize W with AdamOptimizer
            for epoch in range(self._training_epochs):
                self._trainer(self.bound_model, optimizer=optimizer,
                              criterion=self._criterion, epoch=epoch, callback=callback)

            # step 2: update Z, U
            # Z_i^{k+1} = projection(W_i^{k+1} + U_i^k)
            # U_i^{k+1} = U^k + W_i^{k+1} - Z_i^{k+1}
            for i, wrapper in enumerate(self.get_modules_wrapper()):
                z = wrapper.module.weight.data + U[i]
                Z[i] = self._projection(z, wrapper.config['sparsity'], wrapper)
                U[i] = U[i] + wrapper.module.weight.data - Z[i]

        # apply prune
        self.update_mask()

        logger.info('Compression finished.')

        return self.bound_model


class SlimPruner(StructuredFilterPruner, IterativePruner):
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
        super().__init__(model, config_list=config_list, optimizer=optimizer, pruning_algorithm='slim')
        self.training_aware = True
        self.training_epochs = training_epochs
        self.scale = scale
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


class TaylorFOWeightFilterPruner(IterativePruner, StructuredFilterPruner):
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


class ActivationAPoZRankFilterPruner(IterativePruner, StructuredFilterPruner):
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


class ActivationMeanRankFilterPruner(IterativePruner, StructuredFilterPruner):
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






