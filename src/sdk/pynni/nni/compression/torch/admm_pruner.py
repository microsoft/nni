# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
import math
import copy
import csv
import json
import torch
import numpy as np
from schema import And, Optional

from nni.utils import OptimizeMode

from .compressor import Pruner, LayerInfo
from .weight_rank_filter_pruners import L1FilterPruner
from .utils import CompressorSchema
from .utils import get_layers_no_dependency


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


class ADMMPruner(Pruner):
    """
    This is a Pytorch implementation of ADMM Pruner algorithm.

    """

    def __init__(self, model, config_list, trainer, optimize_iterations=30, experiment_data_dir='./'):
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
        experiment_data_dir : string
            PATH to save experiment data
        """
        super().__init__(model, config_list)

        self._trainer = trainer

        self._optimize_iterations = optimize_iterations

        self._experiment_data_dir = experiment_data_dir

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

    def projection(self, weight_arr, percent=10):
        pcen = np.percentile(abs(weight_arr), percent)
        print("percentile " + str(pcen))
        under_threshold = abs(weight_arr) < pcen
        weight_arr[under_threshold] = 0
        return weight_arr

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
            z = wrapper.module.weight.data.copy()  # check , numpy matrix or tensor ?
            Z.append(z)
            U.append(np.zeros_like(z))

        for k in range(self._optimize_iterations):
            print('Iteration %d', k)

            # step 1: optimize W with AdamOptimizer
            optimizer = torch.nn.optimize.AdamOptimizer(
                self.bound_model.parameters(), lr=1e-3)

            # TODO: maybe need to define a customized loss
            # TODO: cross entropy plus sth
            criterion = torch.nn.CrossEntropyLoss()
            for _ in range(5000):
                self._trainer(self.bound_model, optimizer=optimizer,
                              criterion=criterion)

            # step 2: update Z, U
            # Z^{k+1} = projection(W_{k+1} + U^k)
            # U^{k+1} = U^k + W_{k+1} - Z^{k+1}

            # TODO: save sparsity in init
            sparsity = 0

            for idx, wrapper in enumerate(self.get_modules_wrapper()):
                z = wrapper.module.weight.data.copy()  # check , numpy matrix or tensor ?
                Z[idx] = self._projection(
                    wrapper.module.weight.data + U[idx], sparsity)
                Z.append(z)
                U.append(np.zeros_like(z))

        # apply prune

        _logger.info('----------Compression finished--------------')

        return self.bound_model
