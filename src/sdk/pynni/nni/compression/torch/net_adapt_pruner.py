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
# from .weight_rank_filter_pruners import L1FilterPruner
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
        self._model_to_prune = copy.deepcopy(model)

        super().__init__(model, config_list)

        self._fine_tuner = fine_tuner
        self._evaluator = evaluator
        self._optimize_mode = OptimizeMode(optimize_mode)

        # hyper parameters for NetAdapt algorithm
        self._delta_sparsity = 0.1
        self._get_delta_num_weights()

        # overall pruning rate
        self._sparsity = config_list[0]['sparsity']

        # config_list
        self._config_list = []

        self._experiment_data_dir = experiment_data_dir

    def _get_delta_num_weights(self):
        num_weights = []
        for wrapper in self.get_modules_wrapper():
            num_weights.append(wrapper.module.weight.data.numel())

        self._delta_num_weights = self._delta_sparsity * sum(num_weights)

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
        self.modules_wrapper = modules_wrapper

    def calc_mask(self, wrapper, **kwargs):
        return None

    def _update_config_list(self, config_list, op_name, sparsity):
        '''
        update sparsity of op_name
        '''
        config_list_updated = copy.deepcopy(config_list)
        for idx, (_, op_names) in enumerate(config_list_updated):
            if op_name in op_names:
                config_list_updated[idx]['sparsity'] = sparsity
                return config_list_updated

        # if op_name is not in self._config_list, create a new json item
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
        current_sparsity = 1

        # stop condition
        while current_sparsity > self._sparsity:
            _logger.info('Pruning iteration: %d', pruning_iteration)

            # calculate target sparsity of this round
            target_sparsity = current_sparsity - self._delta_sparsity

            # lists to store testing result of pruning different layers
            op_names = []
            performances = []
            sparsities = []

            for wrapper in self.get_modules_wrapper():
                # Choose Num Filters to prune
                # TODO: check related layers
                weight_mask = wrapper.weight_mask

                # sparsity of this layer needs to satisfy the requirement
                sparsity = (weight_mask.sum().item() -
                            self._delta_num_weights) / weight_mask.numel()

                if sparsity <= 0:
                    _logger.info(
                        'This layer has no enough weights remained to prune')
                    continue

                # Choose which filter to prune (l1)
                config_list = self._update_config_list(
                    self._config_list, wrapper.name, sparsity)
                pruner = LevelPruner(copy.deepcopy(
                    self._model_to_prune), config_list)
                model_masked = pruner.compress()

                # Short-term fine tune the pruned model
                self._fine_tuner(model_masked)

                perf = self.evaluator(model_masked)

                op_names.append(wrapper.name)
                sparsities.append(sparsity)
                performances.append(perf)

            # 3. Pick the best layer to prune
            if not performances:
                _logger.info("No more layers to prune.")
                break

            if self._optimize_mode is OptimizeMode.Minimize:
                latest_performance = min(performances)
            else:
                latest_performance = max(performances)
            idx = performances.index(latest_performance)

            self._config_list = self._update_config_list(
                self._config_list, sparsities[idx], op_names[idx])

            current_sparsity = target_sparsity
            _logger.info('Pruning iteration %d, layer %s seleted with additional sparsity %s pruned, latest performance : %s, current overall sparsity : %s',
                         pruning_iteration, op_names[idx], sparsities[idx], latest_performance, current_sparsity)
            pruning_iteration += 1

        self._final_performance = latest_performance
        _logger.info('----------Compression finished--------------')
        _logger.info('config_list generated: %s', self._config_list)

        # save best config found and best performance
        with open(os.path.join(self._experiment_data_dir, 'search_result.json'), 'w') as jsonfile:
            json.dump({
                'performance': self._final_performance,
                'config_list': json.dumps(self._config_list)
            }, jsonfile)

        _logger.info('search history and result saved to foler : %s',
                     self._experiment_data_dir)

        return self.bound_model
