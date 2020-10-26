# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
import math
import copy
import csv
import json
import numpy as np
from schema import And, Optional

from nni.utils import OptimizeMode

from nni.compression.pytorch.compressor import Pruner
from nni.compression.pytorch.utils.config_validation import CompressorSchema
from .constants_pruner import PRUNER_DICT


_logger = logging.getLogger(__name__)


class SimulatedAnnealingPruner(Pruner):
    """
    A Pytorch implementation of Simulated Annealing compression algorithm.

    Parameters
    ----------
    model : pytorch model
        The model to be pruned.
    config_list : list
        Supported keys:
            - sparsity : The target overall sparsity.
            - op_types : The operation type to prune.
    evaluator : function
        Function to evaluate the pruned model.
        This function should include `model` as the only parameter, and returns a scalar value.
        Example::

            def evaluator(model):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                val_loader = ...
                model.eval()
                correct = 0
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        # get the index of the max log-probability
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
                accuracy = correct / len(val_loader.dataset)
                return accuracy
    optimize_mode : str
        Optimize mode, `maximize` or `minimize`, by default `maximize`.
    base_algo : str
        Base pruning algorithm. `level`, `l1` or `l2`, by default `l1`. Given the sparsity distribution among the ops,
        the assigned `base_algo` is used to decide which filters/channels/weights to prune.
    start_temperature : float
        Start temperature of the simulated annealing process.
    stop_temperature : float
        Stop temperature of the simulated annealing process.
    cool_down_rate : float
        Cool down rate of the temperature.
    perturbation_magnitude : float
        Initial perturbation magnitude to the sparsities. The magnitude decreases with current temperature.
    experiment_data_dir : string
        PATH to save experiment data,
        including the config_list generated for the base pruning algorithm, the performance of the pruned model and the pruning history.

    """

    def __init__(self, model, config_list, evaluator, optimize_mode='maximize', base_algo='l1',
                 start_temperature=100, stop_temperature=20, cool_down_rate=0.9, perturbation_magnitude=0.35, experiment_data_dir='./'):
        # original model
        self._model_to_prune = copy.deepcopy(model)
        self._base_algo = base_algo

        super().__init__(model, config_list)

        self._evaluator = evaluator
        self._optimize_mode = OptimizeMode(optimize_mode)

        # hyper parameters for SA algorithm
        self._start_temperature = start_temperature
        self._current_temperature = start_temperature
        self._stop_temperature = stop_temperature
        self._cool_down_rate = cool_down_rate
        self._perturbation_magnitude = perturbation_magnitude

        # overall pruning rate
        self._sparsity = config_list[0]['sparsity']
        # pruning rates of the layers
        self._sparsities = None

        # init current performance & best performance
        self._current_performance = -np.inf
        self._best_performance = -np.inf
        self._best_config_list = []

        self._search_history = []

        self._experiment_data_dir = experiment_data_dir
        if not os.path.exists(self._experiment_data_dir):
            os.makedirs(self._experiment_data_dir)

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
        elif self._base_algo in ['l1', 'l2']:
            schema = CompressorSchema([{
                'sparsity': And(float, lambda n: 0 < n < 1),
                'op_types': ['Conv2d'],
                Optional('op_names'): [str]
            }], model, _logger)

        schema.validate(config_list)

    def _sparsities_2_config_list(self, sparsities):
        '''
        convert sparsities vector into config_list for LevelPruner or L1FilterPruner

        Parameters
        ----------
        sparsities : list
            list of sparsities

        Returns
        -------
        list of dict
            config_list for LevelPruner or L1FilterPruner
        '''
        config_list = []

        sparsities = sorted(sparsities)
        self.modules_wrapper = sorted(
            self.modules_wrapper, key=lambda wrapper: wrapper.module.weight.data.numel())

        # a layer with more weights will have no less pruning rate
        for idx, wrapper in enumerate(self.get_modules_wrapper()):
            # L1Filter Pruner requires to specify op_types
            if self._base_algo in ['l1', 'l2']:
                config_list.append(
                    {'sparsity': sparsities[idx], 'op_types': ['Conv2d'], 'op_names': [wrapper.name]})
            elif self._base_algo == 'level':
                config_list.append(
                    {'sparsity': sparsities[idx], 'op_names': [wrapper.name]})

        config_list = [val for val in config_list if not math.isclose(val['sparsity'], 0, abs_tol=1e-6)]

        return config_list

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
        if total_weights_pruned == 0:
            return None
        scale = target_sparsity / (total_weights_pruned/total_weights)

        # rescale the sparsities
        sparsities = np.asarray(sparsities)*scale

        return sparsities

    def _init_sparsities(self):
        '''
        Generate a sorted sparsities vector
        '''
        # repeatedly generate a distribution until satisfies the overall sparsity requirement
        _logger.info('Gererating sparsities...')
        while True:
            sparsities = sorted(np.random.uniform(
                0, 1, len(self.get_modules_wrapper())))

            sparsities = self._rescale_sparsities(
                sparsities, target_sparsity=self._sparsity)

            if sparsities is not None and sparsities[0] >= 0 and sparsities[-1] < 1:
                _logger.info('Initial sparsities generated : %s', sparsities)
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
        _logger.info("Gererating perturbations to the current sparsities...")

        # decrease magnitude with current temperature
        magnitude = self._current_temperature / \
            self._start_temperature * self._perturbation_magnitude
        _logger.info('current perturation magnitude:%s', magnitude)

        while True:
            perturbation = np.random.uniform(-magnitude, magnitude, len(self.get_modules_wrapper()))
            sparsities = np.clip(0, self._sparsities + perturbation, None)
            _logger.debug("sparsities before rescalling:%s", sparsities)

            sparsities = self._rescale_sparsities(sparsities, target_sparsity=self._sparsity)
            _logger.debug("sparsities after rescalling:%s", sparsities)

            if sparsities is not None and sparsities[0] >= 0 and sparsities[-1] < 1:
                _logger.info("Sparsities perturbated:%s", sparsities)
                return sparsities

    def calc_mask(self, wrapper, **kwargs):
        return None

    def compress(self, return_config_list=False):
        """
        Compress the model with Simulated Annealing.

        Returns
        -------
        torch.nn.Module
            model with specified modules compressed.
        """
        _logger.info('Starting Simulated Annealing Compression...')

        # initiaze a randomized action
        pruning_iteration = 0
        self._init_sparsities()

        # stop condition
        self._current_temperature = self._start_temperature
        while self._current_temperature > self._stop_temperature:
            _logger.info('Pruning iteration: %d', pruning_iteration)
            _logger.info('Current temperature: %d, Stop temperature: %d',
                         self._current_temperature, self._stop_temperature)
            while True:
                # generate perturbation
                sparsities_perturbated = self._generate_perturbations()
                config_list = self._sparsities_2_config_list(
                    sparsities_perturbated)
                _logger.info(
                    "config_list for Pruner generated: %s", config_list)

                # fast evaluation
                pruner = PRUNER_DICT[self._base_algo](copy.deepcopy(self._model_to_prune), config_list)
                model_masked = pruner.compress()
                evaluation_result = self._evaluator(model_masked)

                self._search_history.append(
                    {'sparsity': self._sparsity, 'performance': evaluation_result, 'config_list': config_list})

                if self._optimize_mode is OptimizeMode.Minimize:
                    evaluation_result *= -1

                # if better evaluation result, then accept the perturbation
                if evaluation_result > self._current_performance:
                    self._current_performance = evaluation_result
                    self._sparsities = sparsities_perturbated

                    # save best performance and best params
                    if evaluation_result > self._best_performance:
                        _logger.info('updating best model...')
                        self._best_performance = evaluation_result
                        self._best_config_list = config_list

                        # save the overall best masked model
                        self.bound_model = model_masked
                        # the ops with sparsity 0 are not included in this modules_wrapper
                        modules_wrapper_final = pruner.get_modules_wrapper()
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
            pruning_iteration += 1

        _logger.info('----------Compression finished--------------')
        _logger.info('Best performance: %s', self._best_performance)
        _logger.info('config_list found : %s',
                     self._best_config_list)

        # save search history
        with open(os.path.join(self._experiment_data_dir, 'search_history.csv'), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['sparsity', 'performance', 'config_list'])
            writer.writeheader()
            for item in self._search_history:
                writer.writerow({'sparsity': item['sparsity'], 'performance': item['performance'], 'config_list': json.dumps(
                    item['config_list'])})

        # save best config found and best performance
        if self._optimize_mode is OptimizeMode.Minimize:
            self._best_performance *= -1
        with open(os.path.join(self._experiment_data_dir, 'search_result.json'), 'w+') as jsonfile:
            json.dump({
                'performance': self._best_performance,
                'config_list': json.dumps(self._best_config_list)
            }, jsonfile)

        _logger.info('search history and result saved to foler : %s',
                     self._experiment_data_dir)

        if return_config_list:
            return self._best_config_list

        # This should be done only at the final stage,
        # because the modules_wrapper with all the ops are used during the annealing process
        self.modules_wrapper = modules_wrapper_final

        return self.bound_model
