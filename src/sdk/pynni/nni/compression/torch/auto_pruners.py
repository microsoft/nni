# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import math
import copy
import numpy as np
import torch
from schema import And, Optional


from nni.utils import OptimizeMode

from .compressor import Pruner
from .pruners import LevelPruner
from .utils import CompressorSchema


__all__ = ['SimulatedAnnealingPruner']

logger = logging.getLogger(__name__)


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

    def __init__(self, model, config_list, evaluater, optimize_mode='maximize', T=100, stop_temperature=20, cool_down_rate=0.9):
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

        self._model_to_prune = copy.deepcopy(model)
        self._model_pruned = copy.deepcopy(model)
        self._model_current = copy.deepcopy(model)

        super().__init__(model, config_list)

        self._optimize_mode = OptimizeMode(optimize_mode)

        # hyper parameters of SA algorithm
        self.T = T
        self.stop_temperature = stop_temperature
        self.cool_down_rate = cool_down_rate  # cool down rate

        self._optimize_mode = OptimizeMode(optimize_mode)

        # init pruning rates of the layers
        self.sparsities = self._generate_sparsities()

        # init current performance & best performance
        self.current_performance = -np.inf
        self.best_performance = -np.inf

        self.evaluater = evaluater

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
        sparsities = np.random.uniform(0, 1, len(self.get_modules_wrapper()))
        # TODO: rescale
        # sparsities = self._rescale_sparsities(
        #     self.model, sparsities, config_list['sparsity'])
        # check sparsities

        return sparsities

    def _generate_perturbations(self):
        # TODO: decrease magnitude
        perturbation = np.random.uniform(-0.1, 0.1, self.dimension)
        sparsities_perturbated = np.clip(self.sparsities + perturbation, 0, 1)
        # TODO: check_sparsities(sparsities, model)

        return sparsities_perturbated

    def _rescale_sparsities(self, model, sparsities, target_sparsity):
        # TODO: calculate sparsity and rescale sparsities
        # TODO: sort the sparsities
        return sparsities

    def _generate_config_list_level(self, sparsities):
        '''
        Generate config_list for LevelPruner

        Parameters
        ----------
        sparsities : numpy array
            pruning sparsities of the layers to be pruned

        Returns
        -------
        list of dict
            config_list for LevelPruner, specified by op_names (level)
        '''
        # TODO: check order & sparsities param
        config_list_level = []
        for idx, wrapper in enumerate(self.get_modules_wrapper()):
            config_list_level.append(
                {'sparsity': sparsities[idx], 'op_names': [wrapper.name]})

        return config_list_level

    def compress(self):
        logger.info('Starting Simulated Annealing Compression...')
        logger.info('Current temperature: %d, Stop temperature: %d',
                    self.T, self.stop_temperature)

        if self.T <= self.stop_temperature:
            logger.info('Compression finished')
            # TODO: if SimulatedAnnealingTuner is used seperatedly, return the overall best sparsities
            # else return current sparsities
            # check bound_model
            return self.bound_model

        while True:
            # generate perturbation
            sparsities_perturbated = self._generate_sparsities()
            config_list_level = self._generate_config_list_level(
                sparsities_perturbated)
            logger.info("config_list for LevelPruner generated: %s",
                        config_list_level)

            # fast evaluation
            # TODO: check model parameter
            logger.info("model:%s", self._model_to_prune)
            level_pruner = LevelPruner(
                model=self._model_to_prune, config_list=config_list_level)
            level_pruner.compress()

            level_pruner.export_model('model.pth', 'mask.pth')
            self._model_pruned.load_state_dict(torch.load('model.pth'))
            evaluation_result = self.evaluater(self._model_pruned)

            # if better evaluation result, then accept the perturbation
            if self._optimize_mode is OptimizeMode.Minimize:
                evaluation_result *= -1
            if evaluation_result > self.current_performance:
                self.current_performance = evaluation_result
                self.sparsities = sparsities_perturbated
                self._model_current = self._model_pruned
                # save best performance and best params
                if evaluation_result > self.best_performance:
                    self.best_performance = evaluation_result
                    self.best_sparsities = sparsities_perturbated
                    self._model_best = self._model_pruned
                break
            # if not, accept with probability e^(-deltaE/T)
            else:
                delta_E = np.abs(evaluation_result - self.current_performance)
                probability = math.exp(-1 * delta_E / self.T)
                if np.random.uniform(0, 1) < probability:
                    self.current_performance = evaluation_result
                    self.sparsities = sparsities_perturbated
                    self._model_current = self._model_pruned

        self.T *= self.cool_down_rate
