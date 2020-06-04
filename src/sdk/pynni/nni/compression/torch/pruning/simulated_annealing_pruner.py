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

from ..compressor import Pruner, LayerInfo
from ..utils.config_validation import CompressorSchema
from ..utils.op_dependency import get_layers_no_dependency
from .pruners import LevelPruner
from .weight_rank_filter_pruners import L1FilterPruner


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


class SimulatedAnnealingPruner(Pruner):
    """
    This is a Pytorch implementation of Simulated Annealing compression algorithm.

    - Randomly initialize a pruning rate distribution (sparsities).
    - While current_temperature < stop_temperature:
        1. generate a perturbation to current distribution
        2. Perform fast evaluation on the perturbated distribution
        3. accept the perturbation according to the performance and probability, if not accepted, return to step 1
        4. cool down, current_temperature <- current_temperature * cool_down_rate
    """

    def __init__(self, model, config_list, evaluator, optimize_mode='maximize', pruning_mode='channel',
                 start_temperature=100, stop_temperature=20, cool_down_rate=0.9, perturbation_magnitude=0.35, experiment_data_dir='./'):
        """
        Parameters
        ----------
        model : pytorch model
            The model to be pruned
        config_list : list
            Supported keys:
                - sparsity : The final sparsity when the compression is done.
                - op_types : The operation type to prune.
        evaluator : function
            function to evaluate the pruned model
        optimize_mode : str
            optimize mode, 'maximize' or 'minimize', by default 'maximize'
        pruning_mode : str
            'channel' or 'fine_grained, by default 'channel'
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
        # original model
        self._model_to_prune = copy.deepcopy(model)
        self._pruning_mode = pruning_mode

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

    def _detect_modules_to_compress(self):
        """
        redefine this function, consider only the layers without dependencies
        """
        if self.modules_to_compress is None:
            self.modules_to_compress = []
            # consider only the layers without dependencies
            model_name = self._model_to_prune.__class__.__name__
            ops_no_dependency = get_layers_no_dependency(model_name)

            for name, module in self.bound_model.named_modules():
                if module == self.bound_model:
                    continue
                if self._pruning_mode == 'channel' and model_name in ['MobileNetV2', 'RetinaFace'] and name not in ops_no_dependency:
                    continue
                layer = LayerInfo(name, module)
                config = self.select_config(layer)
                if config is not None:
                    self.modules_to_compress.append((layer, config))
        return self.modules_to_compress

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
            if self._pruning_mode == 'channel':
                config_list.append(
                    {'sparsity': sparsities[idx], 'op_types': ['Conv2d'], 'op_names': [wrapper.name]})
            elif self._pruning_mode == 'fine_grained':
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
            perturbation = np.random.uniform(-magnitude,
                                             magnitude, len(self.get_modules_wrapper()))
            sparsities = np.clip(0, self._sparsities + perturbation, None)
            _logger.debug("sparsities before rescalling:%s", sparsities)

            sparsities = self._rescale_sparsities(
                sparsities, target_sparsity=self._sparsity)
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
                if self._pruning_mode == 'channel':
                    pruner = L1FilterPruner(
                        model=copy.deepcopy(self._model_to_prune), config_list=config_list)
                elif self._pruning_mode == 'fine_grained':
                    pruner = LevelPruner(
                        model=copy.deepcopy(self._model_to_prune), config_list=config_list)
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

        return self.bound_model
