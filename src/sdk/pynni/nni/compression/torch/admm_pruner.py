# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import copy
import torch
from schema import And, Optional

from .compressor import Pruner
from .pruners import LevelPruner
from .weight_rank_filter_pruners import L1FilterPruner
from .utils import CompressorSchema


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


class ADMMPruner(Pruner):
    """
    This is a Pytorch implementation of ADMM Pruner algorithm.
    For details, please refer to paper https://arxiv.org/abs/1804.03294.
    """

    def __init__(self, model, config_list, trainer, optimize_iterations=30, epochs=5, pruning_mode='fine_grained'):
        """
        Parameters
        ----------
        model : pytorch model
            The model to be pruned
        config_list : list
            Supported keys:
                - sparsity : The final sparsity when the compression is done.
                - op_names : The operation type to prune.
        trainer : function
            function used for the first step of ADMM training
        optimize_iteration : int
            ADMM optimize iterations
        pruning_mode : str
            'channel' or 'fine_grained, by default 'fine_grained'
        """
        self._pruning_mode = pruning_mode
        # TODO: modules to compress

        super().__init__(model, config_list)

        self._trainer = trainer
        self._optimize_iterations = optimize_iterations
        self._epochs = 5

        self.set_wrappers_attribute("if_calculated", False)

    def validate_config(self, model, config_list):
        """
        Parameters
        ----------
        model : torch.nn.module
            Model to be pruned
        config_list : list
            Supported keys:
                - prune_iterations : The number of rounds for the iterative pruning.
                - sparsity : The final sparsity when the compression is done.
        """
        schema = CompressorSchema([{
            'sparsity': And(float, lambda n: 0 < n < 1),
            Optional('op_types'): [str],
            Optional('op_names'): [str],
        }], model, _logger)

        schema.validate(config_list)

    # TODO: check pruning mode
    def _projection(self, weight, sparsity=0.1):
        w_abs = weight.abs()
        k = int(weight.numel() * sparsity)
        if k == 0:
            mask_weight = torch.ones(weight.shape).type_as(weight)
        else:
            threshold = torch.topk(w_abs.view(-1), k, largest=False)[0].max()
            mask_weight = torch.gt(w_abs, threshold).type_as(weight)

        return weight.data.mul(mask_weight)

    def calc_mask(self, wrapper, **kwargs):
        if self._pruning_mode == 'fine_grained':
            return LevelPruner.calc_mask(self, wrapper, **kwargs)
        elif self._pruning_mode == 'channel':
            return L1FilterPruner.calc_mask(self, wrapper, **kwargs)

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
        # Z^0 = W_0
        # U^0 = 0
        Z = []
        U = []
        for wrapper in self.get_modules_wrapper():
            z = wrapper.module.weight.data
            Z.append(z)
            U.append(torch.zeros_like(z))

        # TODO: define a customized loss / optimizer
        optimizer = torch.optim.Adam(
            self.bound_model.parameters(), lr=1e-3, weight_decay=5e-5)

        # Loss = cross_entropy +  l2 regulization + \Sum_{i=1}^N \row_i ||W_i - Z_i^k + U_i^k||^2
        criterion = torch.nn.CrossEntropyLoss()

        # callback function to do additonal optimization, refer to Formula (7)
        def callback():
            for i, wrapper in enumerate(self.get_modules_wrapper()):
                row = 1e-4
                wrapper.module.weight.data -= row * \
                    (wrapper.module.weight.data - Z[i] + U[i])

        # optimization iteration
        for k in range(self._optimize_iterations):
            print('Iteration %d', k)

            # step 1: optimize W with AdamOptimizer
            for _ in range(self._epochs):
                self._trainer(self.bound_model, optimizer=optimizer,
                              criterion=criterion, callback=callback)

            # step 2: update Z, U
            # Z_i^{k+1} = projection(W_i^{k+1} + U_i^k)
            # U_i^{k+1} = U^k + W_i^{k+1} - Z_i^{k+1}
            for i, wrapper in enumerate(self.get_modules_wrapper()):
                z = wrapper.module.weight.data + U[i]
                Z[i] = self._projection(
                    z, wrapper.config['sparsity'])
                U[i] = U[i] + wrapper.module.weight.data - Z[i]

            # TODO: check stop conditions : formula (6) .. not necessary

        # apply prune
        self.update_mask()

        _logger.info('----------Compression finished--------------')

        return self.bound_model
