# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
import copy
import json
from schema import And, Optional

from nni.utils import OptimizeMode

from .compressor import Pruner
from .pruners import LevelPruner
from .weight_rank_filter_pruners import L1FilterPruner
from .utils import CompressorSchema


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


class NetAdaptPruner(Pruner):
    """
    This is a Pytorch implementation of NetAdapt compression algorithm.

    The pruning procedure can be described as follows:
    While Res_i > Bud:
        1. Con = Res_i - delta_Res
        2. for every layer:
            Choose Num Filters to prune
            Choose which filter to prunee (l1)
            Short-term fine tune the pruned model
        3. Pick the best layer to prune
    Long-term fine tune
    """

    def __init__(self, model, config_list, evaluator, fine_tuner, optimize_mode='maximize', pruning_mode='channel', experiment_data_dir='./'):
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
        fine_tuner : function
            function to fine tune the pruned model
        optimize_mode : str
            optimize mode, 'maximize' or 'minimize', by default 'maximize'
        pruning_mode : str
            'channel' or 'fine_grained, by default 'channel'
        experiment_data_dir : str
            PATH to save experiment data
        """
        # models used for iterative pruning and evaluation
        self._original_model = copy.deepcopy(model)

        super().__init__(model, config_list)

        self._fine_tuner = fine_tuner
        self._evaluator = evaluator
        self._optimize_mode = OptimizeMode(optimize_mode)
        self._pruning_mode = pruning_mode

        # hyper parameters for NetAdapt algorithm
        self._delta_sparsity = 0.1

        # overall pruning rate
        self._sparsity = config_list[0]['sparsity']

        # config_list
        self._config_list = []

        self._experiment_data_dir = experiment_data_dir

    def _get_delta_num_weights(self, iteration):
        num_weights = []
        for wrapper in self.get_modules_wrapper():
            num_weights.append(wrapper.module.weight.data.numel())

        delta_num_weights = self._delta_sparsity * \
            sum(num_weights) * (iteration+1)

        return delta_num_weights

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

    def _set_modules_wrapper(self, modules_wrapper):
        """
        To obtain all the wrapped modules.

        Parameters
        -------
        list
            a list of the wrapped modules
        """
        self.modules_wrapper = copy.deepcopy(modules_wrapper)

    def calc_mask(self, wrapper, **kwargs):
        return None

    def _update_config_list(self, config_list, op_name, sparsity):
        '''
        update sparsity of op_name in config_list
        '''
        config_list_updated = copy.deepcopy(config_list)
        for idx, item in enumerate(config_list):
            if op_name in item['op_names']:
                config_list_updated[idx]['sparsity'] = sparsity
                return config_list_updated

        # if op_name is not in self._config_list, create a new json item
        if self._pruning_mode == 'channel':
            config_list_updated.append(
                {'sparsity': sparsity, 'op_types': ['Conv2d'], 'op_names': [op_name]})
        elif self._pruning_mode == 'fine_grained':
            config_list_updated.append(
                {'sparsity': sparsity, 'op_names': [op_name]})
        return config_list_updated

    def compress(self):
        """
        Compress the model.

        Returns
        -------
        torch.nn.Module
            model with specified modules compressed.
        """
        _logger.info('Starting NetAdapt Compression...')

        pruning_iteration = 0
        current_sparsity = 0

        # stop condition
        while current_sparsity < self._sparsity:
            _logger.info('Pruning iteration: %d', pruning_iteration)

            # calculate target sparsity of this round
            target_sparsity = current_sparsity + self._delta_sparsity
            delta_num_weights = self._get_delta_num_weights(pruning_iteration)

            # variable store the best result of one iteration
            best_layer = {}

            for wrapper in self.get_modules_wrapper():
                # Choose Num Filters to prune
                # TODO: check related layers
                # sparsity that this layer needs to prune to satisfy the requirement
                sparsity = delta_num_weights / wrapper.weight_mask.numel()

                if sparsity >= 1:
                    _logger.info(
                        'Layer %s has no enough weights remained to prune', wrapper.name)
                    continue

                # Choose which filter to prune (l1)
                config_list = self._update_config_list(
                    self._config_list, wrapper.name, sparsity)
                _logger.info("config_list used : %s", config_list)

                if self._pruning_mode == 'channel':
                    pruner = L1FilterPruner(copy.deepcopy(
                        self._original_model), config_list)
                elif self._pruning_mode == 'fine_grained':
                    pruner = LevelPruner(copy.deepcopy(
                        self._original_model), config_list)
                model_masked = pruner.compress()

                performance = self._evaluator(model_masked)
                _logger.info(
                    "Layer : %s, evaluation result before fine tuning : %s", wrapper.name, performance)
                # Short-term fine tune the pruned model
                self._fine_tuner(model_masked)

                performance = self._evaluator(model_masked)
                _logger.info(
                    "Layer : %s, evaluation result after short-term fine tuning : %s", wrapper.name, performance)

                if not best_layer or (self._optimize_mode is OptimizeMode.Maximize and performance > best_layer['performance']) or (self._optimize_mode is OptimizeMode.Minimize and performance < best_layer['performance']):
                    _logger.debug("updating best layer to %s...", wrapper.name)
                    best_layer = {
                        'op_name': wrapper.name,
                        'sparsity': sparsity,
                        'performance': performance
                    }
                    # update bound model
                    # TODO: why set modules_wrapper doesn't work ?
                    self.bound_model = model_masked
                    # self._set_modules_wrapper(pruner.get_modules_wrapper())
                    # self._wrap_model()

            # 3. Pick the best layer to prune
            if not best_layer:
                _logger.info("No more layers to prune.")
                break

            self._config_list = self._update_config_list(
                self._config_list, best_layer['op_name'], best_layer['sparsity'])

            current_sparsity = target_sparsity
            _logger.info('Pruning iteration %d finished. Layer %s seleted with sparsity %s, performance after pruning & short term fine-tuning : %s, current overall sparsity : %s',
                         pruning_iteration, best_layer['op_name'], best_layer['sparsity'], best_layer['performance'], current_sparsity)
            pruning_iteration += 1

            # update bound model after each iteration
            # _logger.debug("Updating bound model...")
            # self._set_modules_wrapper(best_layer['modules_wrapper'])
            # self._wrap_model()
            self._final_performance = best_layer['performance']

        _logger.info('----------Compression finished--------------')
        _logger.info('config_list generated: %s', self._config_list)

        _logger.debug("Performance (bound model): %s",
                      self._evaluator(self.bound_model))

        # save best config found and best performance
        if not os.path.exists(self._experiment_data_dir):
            os.makedirs(self._experiment_data_dir)
        with open(os.path.join(self._experiment_data_dir, 'search_result.json'), 'w') as jsonfile:
            json.dump({
                'performance': self._final_performance,
                'config_list': json.dumps(self._config_list)
            }, jsonfile)

        _logger.info('search history and result saved to foler : %s',
                     self._experiment_data_dir)

        return self.bound_model
