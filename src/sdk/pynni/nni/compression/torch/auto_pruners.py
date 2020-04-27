# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import math
import copy
import json
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

    1. Randomly initialize a pruning rate distribution (sparsities).
    2. generate perturbation
    3. Perform fast evaluation on the perturbation
    4. accept the perturbation according to the performance and probability, if not accepted, return to step 2
    5. cool down, current_temperature <- current_temperature * cool_down_rate
    6. repeat step 2~5 while current_temperature > stop_temperature
    """

    def __init__(self, model, config_list, evaluator, optimize_mode='maximize',
                 start_temperature=100, stop_temperature=20, cool_down_rate=0.5, perturbation_magnitude=0.35, experiment_data_dir='./'):
        """
        Parameters
        ----------
        model : pytorch model
            The model to be pruned
        config_list : list
            Supported keys:
                - sparsity : The final sparsity when the compression is done.
                - op_type : The operation type to prune.
        evaluator : function
            function to evaluate the pruned model
        optimize_mode : string
            optimize mode, 'maximize' or 'minimize', by default 'maximize'
        start_temperature : float
            Simualated Annealing related parameter
        stop_temperature : float
            Simualated Annealing related parameter
        cool_down_rate : float
            Simualated Annealing related parameter
        perturbation_magnitude : float
            initial perturbation magnitude to the sparsities. The magnitude decreases with current temperature
        experiment_data_dir : string
            PATH to save experiment data
        """
        # models used for iterative pruning and evaluation
        self._model_to_prune = copy.deepcopy(model)
        self._model_pruned = copy.deepcopy(model)

        self._experiment_data_dir = experiment_data_dir
        self._TMP_MODEL_PATH = '{}model.pth'.format(self._experiment_data_dir)

        super().__init__(model, config_list)

        self._evaluater = evaluator
        self._optimize_mode = OptimizeMode(optimize_mode)

        # hyper parameters for SA algorithm
        self._start_temperature = start_temperature
        self._stop_temperature = stop_temperature
        self._cool_down_rate = cool_down_rate
        self._perturbation_magnitude = perturbation_magnitude

        # pruning rates of the layers
        self._sparsities = None

        # init current performance & best performance
        self._current_performance = -np.inf
        self._best_performance = -np.inf

        self._pruning_iteration = 0
        self._pruning_history = []

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

    def _sparsities_2_config_list_level(self, sparsities):
        '''
        convert sparsities vector into config_list_level for LevelPruner

        Parameters
        ----------
        sparsities : list
            list of sparsities

        Returns
        -------
        list of dict
            config_list_level for LevelPruner
        '''
        config_list_level = []

        sparsities = sorted(sparsities)
        self.modules_wrapper = sorted(
            self.modules_wrapper, key=lambda wrapper: wrapper.module.weight.data.numel())

        # a layer with more weights will have no less pruning rate
        for idx, wrapper in enumerate(self.get_modules_wrapper()):
            config_list_level.append(
                {'sparsity': sparsities[idx], 'op_names': [wrapper.name]})

        return config_list_level

    def _rescale_sparsities(self, sparsities, target_sparsity):
        '''
        Rescale the sparsities list to satisfy the target overall sparsity

        Parameters
        ----------
        sparsities : list

        target_sparsity : float
            the target overall sparsity

        Returns
        -------
        list
            the rescaled sparsities
        '''
        num_weights = []
        for wrapper in self.get_modules_wrapper():
            num_weights.append(wrapper.module.weight.data.numel())

        num_weights = sorted(num_weights)
        sparsities = sorted(sparsities)

        total_weights = 0
        total_weights_pruned = 0

        # calculate the scale
        for idx, num_weight in enumerate(num_weights):
            total_weights += num_weight
            total_weights_pruned += int(num_weight*sparsities[idx])
        scale = target_sparsity / (total_weights_pruned/total_weights)

        # rescale the sparsities
        sparsities = np.asarray(sparsities)*scale

        return sparsities

    def _init_sparsities(self):
        '''
        Generate a sorted sparsities vector
        '''
        # repeatedly generate a distribution until satisfies the overall sparsity requirement
        logger.info('Gererating sparsities...')
        while True:
            sparsities = sorted(np.random.uniform(
                0, 1, len(self.get_modules_wrapper())))

            sparsities = self._rescale_sparsities(
                sparsities, target_sparsity=self.config_list[0]['sparsity'])

            if sparsities[0] >= 0 and sparsities[-1] < 1:
                logger.info('Initial sparsities generated : %s', sparsities)
                self._sparsities = sparsities
                break

    def _generate_perturbations(self):
        '''
        Generate perturbation to the current sparsities distribution.

        Returns:
        --------
        list
            perturbated sparsities
        '''
        logger.info("Gererating perturbations to the current sparsities...")

        # decrease magnitude with current temperature
        magnitude = self._current_temperature / \
            self._start_temperature * self._perturbation_magnitude
        logger.info('current perturation magnitude:%s', magnitude)

        while True:
            perturbation = np.random.uniform(-magnitude,
                                             magnitude, len(self.get_modules_wrapper()))
            sparsities = np.clip(0, self._sparsities + perturbation, None)
            logger.info("sparsities before rescalling:%s", sparsities)

            sparsities = self._rescale_sparsities(
                sparsities, target_sparsity=self.config_list[0]['sparsity'])
            logger.info("sparsities after rescalling:%s", sparsities)

            if sparsities[0] >= 0 and sparsities[-1] < 1:
                logger.info("Sparsities perturbated:%s", sparsities)
                return sparsities

    def _set_modules_wrapper(self, modules_wrapper):
        """
        To obtain all the wrapped modules.

        Parameters
        -------
        list
            a list of the wrapped modules
        """
        self.modules_wrapper = modules_wrapper

    def calc_mask(self, wrapper, **kwargs):
        return None

    def compress(self):
        """
        Compress the model with Simulated Annealing.

        Returns
        -------
        torch.nn.Module
            model with specified modules compressed.
        """
        logger.info('Starting Simulated Annealing Compression...')

        # initiaze a randomized action
        self._init_sparsities()

        # stop condition
        self._current_temperature = self._start_temperature
        while self._current_temperature > self._stop_temperature:
            logger.info('Pruning iteration: %d', self._pruning_iteration)
            logger.info('Current temperature: %d, Stop temperature: %d',
                        self._current_temperature, self._stop_temperature)
            while True:
                # generate perturbation
                sparsities_perturbated = self._generate_perturbations()
                config_list_level = self._sparsities_2_config_list_level(
                    sparsities_perturbated)
                logger.info("config_list for LevelPruner generated: %s",
                            config_list_level)

                # fast evaluation
                level_pruner = LevelPruner(
                    model=copy.deepcopy(self._model_to_prune), config_list=config_list_level)
                level_pruner.compress()

                level_pruner.export_model(self._TMP_MODEL_PATH)
                self._model_pruned.load_state_dict(
                    torch.load(self._TMP_MODEL_PATH))
                evaluation_result = self._evaluater(self._model_pruned)

                self._pruning_history.append(
                    {'performance': evaluation_result, 'config_list': config_list_level, })

                if self._optimize_mode is OptimizeMode.Minimize:
                    evaluation_result *= -1

                # if better evaluation result, then accept the perturbation
                if evaluation_result > self._current_performance:
                    self._current_performance = evaluation_result
                    self._sparsities = sparsities_perturbated
                    # save best performance and best params
                    if evaluation_result > self._best_performance:
                        self._best_performance = evaluation_result
                        # if SimulatedAnnealingTuner is used seperately, return the overall best sparsities
                        # else return current sparsities
                        self._set_modules_wrapper(
                            level_pruner.get_modules_wrapper())
                    break
                # if not, accept with probability e^(-deltaE/current_temperature)
                else:
                    delta_E = np.abs(evaluation_result -
                                     self._current_performance)
                    probability = math.exp(-1 * delta_E /
                                           self._current_temperature)
                    if np.random.uniform(0, 1) < probability:
                        self._current_performance = evaluation_result
                        self._sparsities = sparsities_perturbated
                        break

            # cool down
            self._current_temperature *= self._cool_down_rate
            self._pruning_iteration += 1

        logger.info('----------Compression finished--------------')
        logger.info('Best performance: %s', self._best_performance)
        logger.info('Sparsities generated: %s', self._sparsities)
        logger.info('config_list found for LevelPruner: %s',
                    self._sparsities_2_config_list_level(self._sparsities))

        with open('{}pruning_history.txt'.format(self._experiment_data_dir), 'w') as outfile:
            json.dump(self._pruning_history, outfile)
        logger.info('pruning history saved to pruning_history.txt')

        return self.bound_model
