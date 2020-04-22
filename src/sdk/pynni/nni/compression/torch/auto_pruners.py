# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import math
from schema import And, Optional
import numpy as np

from nni.utils import OptimizeMode

from .compressor import Pruner
from .pruners import LevelPruner
from .utils import CompressorSchema


__all__ = ['SimulatedAnnealingPruner']

logger = logging.getLogger('torch pruner')


class SimulatedAnnealingPruner(Pruner):
    """
    This is a Pytorch implementation of Simulated Annealing compression algorithm.

    1. Randomly initialize a pruning rate distribution.
    2. Generate perturbation
    3. Perform fast evaluation on the perturbation
    4. accept the perturbation with probability
    5. cool down, T <- T * cool_down_rate
    6. repeat step 2~5 while T > stop_temperature
    """

    def __init__(self, model, config_list, optimizer=None, optimize_mode='maximize', T=100, stop_temperature=20, cool_down_rate=0.9):
        """
        Parameters
        ----------
        model : pytorch model
            The model to be pruned
        config_list : list
            Supported keys:
                - sparsity : The final sparsity when the compression is done.
                - op_type : The operation type to prune.

        """
        super().__init__(model, config_list)
        # hyper parameters of SA algorithm
        self.T = T
        self.stop_temperature = stop_temperature
        self.cool_down_rate = cool_down_rate  # cool down rate

        self._optimize_mode = OptimizeMode(optimize_mode)
        self.state = []  # the current pruning rates of the layers
        self.current_performance = None

        # init pruning rates of the layers
        sparsities = np.random.uniform(0, 1, len(self.modules_warpper))
        sparsities = self.rescale_sparsities(
            model, sparsities, config_list['sparsity'])

        # init current performance
        self.current_performance = -np.inf

    def rescale_sparsities(self, model, sparsities, target_sparsity):
        # TODO: calculate sparsity and rescale sparsities
        # TODO: sort the sparsities
        return sparsities

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
        }], model, logger)

        schema.validate(config_list)

    def calc_mask(self, wrapper, **kwargs):
        """
        Generate mask for the given ``weight``.

        Parameters
        ----------
        wrapper : Module
            The layer to be pruned

        Returns
        -------
        tensor
            The mask for this weight, it is ```None``` because this pruner
            calculates and assigns masks in ```prune_iteration_start```,
            no need to do anything in this function.
        """
        return None

    def _generate_sparsities(self):
        # TODO: decrease magnitude
        perturbation = np.random.uniform(-0.1, 0.1, self.dimension)
        self.state = self.state + perturbation
        # TODO: refine
        np.clip(self.state, 0, 1)
        return self.__array_to_params(self.state)

    def compress(self):
        logger.info('Simulated Annealing Compression beginning...')
        logger.info('Current temperature: %d', self.T)
        logger.info('Stop temperature: %d', self.stop_temperature)
        if self.T <= self.stop_temperature:
            logger.info('Compression finished')
            return self.bound_model

        while True:
            # generate perturbation
            sparsities = self._generate_sparsities()
            # TODO: check_sparsities(sparsities, model)

            config_list = self.state
            config_list_level = [{'sparsity': params['conv0_sparsity'], 'op_names': ['conv1']},
                     {'sparsity': params['conv1_sparsity'], 'op_names': ['conv2']}]

            # fast evaluation TODO:check format
            evaluation_result = LevelPruner(self.bound_model, config_list)

            # if better evaluation result, then accept the perturbation
            if self.optimize_mode is OptimizeMode.Minimize:
                evaluation_result *= -1
            if evaluation_result > self.current_performance:
                self.current_performance = evaluation_result
                self.state = self.__params_to_array(sparsities)
                break
                # TODO: save best performance and best params
            # if not, accept with probability exp(-deltaE/T)
            else:
                delta_E = np.abs(evaluation_result - self.current_performance)
                probability = math.exp(-1 * delta_E / self.T)
                if np.random.uniform(0, 1) < probability:
                    self.current_performance = evaluation_result
                    self.state = self.__params_to_array(sparsities)

        self.T *= self.cool_down_rate
