# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import torch
from schema import And, Optional

from ..compressor import Pruner
from ..utils.config_validation import CompressorSchema
from .pruners import LevelPruner
from .weight_rank_filter_pruners import L1FilterPruner


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


class ADMMPruner(Pruner):
    """
    This is a Pytorch implementation of ADMM Pruner algorithm.

    Alternating Direction Method of Multipliers (ADMM) is a mathematical optimization technique,
    by decomposing an original problem into two subproblems that can be solved separately.
    In weight pruning problem, the two subproblems are solved via gradient descent algorithm and Euclidean projection respectively.
    This solution framework applies both to non-structured and different variations of structured pruning schemes.

    For more details, please refer to the paper: https://arxiv.org/abs/1804.03294.
    """

    def __init__(self, model, config_list, trainer, optimize_iterations=30, training_epochs=5, row=1e-4, pruning_mode='channel'):
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
            Function used for the first step of ADMM pruning.
            This function should take a pytorch model, optimizer, criterion, epoch, callback as parameters and train the model,
            no return is required.
        optimize_iterations : int
            ADMM optimize iterations
        training_epochs : int
            training epochs of the first optimization subproblem
        row : float
            penalty parameters for ADMM training
        pruning_mode : str
            'channel' or 'fine_grained, by default 'channel'
        """
        self._pruning_mode = pruning_mode

        super().__init__(model, config_list)

        self._trainer = trainer
        self._optimize_iterations = optimize_iterations
        self._training_epochs = training_epochs
        self._row = row

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

    def _projection(self, weight, sparsity=0.1):
        '''
        Return the Euclidean projection of the weight matrix according to the pruning mode.

        Parameters
        ----------
        weight : tensor
            original matrix
        sparsity : float
            the ratio of parameters which need to be set to zero

        Returns
        -------
        tensor
            the projected matrix
        '''
        w_abs = weight.abs()
        if self._pruning_mode == 'fine_grained':
            k = int(weight.numel() * sparsity)
            if k == 0:
                mask_weight = torch.ones(weight.shape).type_as(weight)
            else:
                threshold = torch.topk(
                    w_abs.view(-1), k, largest=False)[0].max()
                mask_weight = torch.gt(w_abs, threshold).type_as(weight)
        elif self._pruning_mode == 'channel':
            filters = weight.size(0)
            num_prune = int(filters * sparsity)
            if filters < 2 or num_prune < 1:
                mask_weight = torch.ones(
                    weight.size()).type_as(weight).detach()
            else:
                w_abs_structured = w_abs.view(filters, -1).sum(dim=1)
                threshold = torch.topk(w_abs_structured.view(-1),
                                       num_prune, largest=False)[0].max()
                mask_weight = torch.gt(w_abs_structured, threshold)[
                    :, None, None, None].expand_as(weight).type_as(weight)

        return weight.data.mul(mask_weight)

    def get_mask(self, base_mask, weight, num_prune):
        return L1FilterPruner.get_mask(self, base_mask, weight, num_prune)

    def calc_mask(self, wrapper, **kwargs):
        """
        Calculate the mask of given layer.
        Use the function of LevelPruner or L1FilterPruner according to the pruning mode.

        Parameters
        ----------
        wrapper : Module
            the module to instrument the compression operation
        Returns
        -------
        dict
            dictionary for storing masks
        """

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
        for k in range(self._optimize_iterations):
            print('ADMM iteration : ', k)

            # step 1: optimize W with AdamOptimizer
            for epoch in range(self._training_epochs):
                self._trainer(self.bound_model, optimizer=optimizer,
                              criterion=criterion, epoch=epoch, callback=callback)

            # step 2: update Z, U
            # Z_i^{k+1} = projection(W_i^{k+1} + U_i^k)
            # U_i^{k+1} = U^k + W_i^{k+1} - Z_i^{k+1}
            for i, wrapper in enumerate(self.get_modules_wrapper()):
                z = wrapper.module.weight.data + U[i]
                Z[i] = self._projection(
                    z, wrapper.config['sparsity'])
                U[i] = U[i] + wrapper.module.weight.data - Z[i]

        # apply prune
        self.update_mask()

        _logger.info('Compression finished.')

        return self.bound_model

