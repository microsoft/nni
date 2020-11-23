# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import torch
from schema import And, Optional
import copy

from nni.compression.pytorch.utils.config_validation import CompressorSchema
from .constants import MASKER_DICT
from .one_shot import OneshotPruner


_logger = logging.getLogger(__name__)


class ADMMPruner(OneshotPruner):
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

    def __init__(self, model, config_list, trainer, num_iterations=30, training_epochs=5, row=1e-4, base_algo='l1'):
        self._base_algo = base_algo

        super().__init__(model, config_list)

        self._trainer = trainer
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
            }], model, _logger)
        elif self._base_algo in ['l1', 'l2', 'fpgm']:
            schema = CompressorSchema([{
                'sparsity': And(float, lambda n: 0 < n < 1),
                'op_types': ['Conv2d'],
                Optional('op_names'): [str]
            }], model, _logger)

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
        criterion = torch.nn.CrossEntropyLoss()

        # callback function to do additonal optimization, refer to the deriatives of Formula (7)
        def callback():
            for i, wrapper in enumerate(self.get_modules_wrapper()):
                wrapper.module.weight.data -= self._row * \
                    (wrapper.module.weight.data - Z[i] + U[i])

        # optimization iteration
        for k in range(self._num_iterations):
            _logger.info('ADMM iteration : %d', k)

            # step 1: optimize W with AdamOptimizer
            for epoch in range(self._training_epochs):
                self._trainer(self.bound_model, optimizer=optimizer,
                              criterion=criterion, epoch=epoch, callback=callback)

            # step 2: update Z, U
            # Z_i^{k+1} = projection(W_i^{k+1} + U_i^k)
            # U_i^{k+1} = U^k + W_i^{k+1} - Z_i^{k+1}
            for i, wrapper in enumerate(self.get_modules_wrapper()):
                z = wrapper.module.weight.data + U[i]
                Z[i] = self._projection(z, wrapper.config['sparsity'], wrapper)
                U[i] = U[i] + wrapper.module.weight.data - Z[i]

        # apply prune
        self.update_mask()

        _logger.info('Compression finished.')

        return self.bound_model
