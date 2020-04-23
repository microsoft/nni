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

    1. Randomly initialize a pruning rate distribution (sparsities).
    2. generate perturbation
    3. Perform fast evaluation on the perturbation
    4. accept the perturbation according to the performance and probability, if not accepted, return to step 2
    5. cool down, current_temperature <- current_temperature * cool_down_rate
    6. repeat step 2~5 while current_temperature > stop_temperature
    """

    def __init__(self, model, config_list, evaluator, optimize_mode='maximize',
                 start_temperature=100, stop_temperature=20, cool_down_rate=0.5, TMP_MODEL_PATH='model.pth'):
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
        TMP_MODEL_PATH : string
            PATH to save temporary models generated by LevelPruner
        """
        # models used for iterative pruning and evaluation
        self._model_to_prune = copy.deepcopy(model)
        self._model_pruned = copy.deepcopy(model)
        self._TMP_MODEL_PATH = TMP_MODEL_PATH

        super().__init__(model, config_list)

        self._evaluater = evaluator
        self._optimize_mode = OptimizeMode(optimize_mode)

        # hyper parameters for SA algorithm
        self._current_temperature = start_temperature
        self._stop_temperature = stop_temperature
        self._cool_down_rate = cool_down_rate  # cool down rate

        # pruning rates of the layers
        self._sparsities = None

        # init current performance & best performance
        self._current_performance = -np.inf
        self._best_performance = -np.inf

        self._pruning_iteration = 0

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
        schema = CompressorSchema({
            'sparsity': And(float, lambda n: 0 < n < 1),
            Optional('op_types'): [str],
        }, model, logger)

        schema.validate(config_list)

    def _rescale_sparsities(self, config_list_level, target_sparsity):
        '''
        Rescale the config list level to satisfy the target overall sparsity

        Parameters
        ----------
        config_list_level : list of dict
            config_list for LevelPruner, specified by op_names (level)

        target_sparsity : float
            the target overall sparsity

        Returns
        -------
        list of dict
            the rescaled config_list_level
        '''
        op_names = []
        sparsities = []
        for item in config_list_level:
            op_names.append(item['op_names'][0])
            sparsities.append(item['sparsity'])

        total_weights = 0
        total_weights_pruned = 0

        for wrapper in self.get_modules_wrapper():
            op_name = wrapper.module.name
            num_weight = wrapper.module.weight.data.numel()
            total_weights += num_weight

            if op_name in op_names:
                idx = op_names.index(op_name)
                total_weights_pruned += int(num_weight*sparsities[idx])

        scale = target_sparsity / (total_weights_pruned/total_weights)
        for item in config_list_level:
            item['sparsity'] *= scale

        return config_list_level

    def _sparsities_2_config_list_level(self, sparsities):
        '''
        convert sparsities vector into config_list_level for LevelPruner

        Parameters
        ----------
        sparsities : list
            sorted list of sparsities

        Returns
        -------
        list of dict
            the rescaled config_list_level
        '''
        config_list_level = []
        for idx, wrapper in enumerate(self.get_modules_wrapper()):
            config_list_level.append(
                {'sparsity': sparsities[idx], 'op_names': [wrapper.name]})

        config_list_level = self._rescale_sparsities(config_list_level, self.config_list[0]['sparsity'])

        return config_list_level
    
    def _init_config_list_level(self):
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
        # TODO : check 
        # a layer with more weights will have no less pruning rate
        self._sparsities = sorted(np.random.uniform(0, 1, len(self.get_modules_wrapper())))
        self.modules_wrapper = sorted(self.modules_wrapper, key=lambda wrapper: wrapper.module.weight.data.numel())

    def _generate_perturbations(self):
        # TODO: decrease magnitude
        perturbation = np.random.uniform(-0.1, 0.1, len(self.get_modules_wrapper()))
        self._sparsities = np.clip(self._sparsities + perturbation, 0, 1)

        # a layer with more weights will have no less pruning rate
        for idx, sparsity in enumerate(self._sparsities):
            if idx > 0 and sparsity > self._sparsities[idx-1]:
                self._sparsities[idx] = self._sparsities[idx-1]

        config_list_level = self._sparsities_2_config_list_level(self._sparsities)

        return config_list_level

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
        '''
        TODO: This method should not be abstract in class Pruner
        '''
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
        self._init_config_list_level()

        # stop condition
        while self._current_temperature > self._stop_temperature:
            logger.info('Pruning iteration: %d', self._pruning_iteration)
            logger.info('Current temperature: %d, Stop temperature: %d',
                        self._current_temperature, self._stop_temperature)
            while True:
                # generate perturbation
                config_list_level = self._generate_perturbations()
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

                if self._optimize_mode is OptimizeMode.Minimize:
                    evaluation_result *= -1

                # if better evaluation result, then accept the perturbation
                if evaluation_result > self._current_performance:
                    self._current_performance = evaluation_result
                    self._sparsities = sparsities_perturbated
                    # save best performance and best params
                    if evaluation_result > self._best_performance:
                        self._best_performance = evaluation_result
                        # if SimulatedAnnealingTuner is used seperatedly, return the overall best sparsities
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

        logger.info('Compression finished')
        logger.info('Best performance: %s', self._best_performance)
        logger.info('Sparsities generated: %s', self._sparsities)

        return self.bound_model
