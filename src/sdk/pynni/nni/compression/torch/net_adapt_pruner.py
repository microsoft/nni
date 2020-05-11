# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
import copy
import csv
import json
import numpy as np
from schema import And, Optional

from nni.utils import OptimizeMode

from .compressor import Pruner
from .pruners import LevelPruner
from .weight_rank_filter_pruners import L1FilterPruner
from .utils import CompressorSchema


__all__ = ['NetAdaptPruner']

_logger = logging.getLogger(__name__)


class NetAdaptPruner(Pruner):
    """
    This is a Pytorch implementation of NetAdapt compression algorithm.

    While Res_i > Bud:
        1. Con = Res_i - delta_Res
        2. for every layer:
            Choose Num Filters to prune
            Choose which filter to prunee (l1)
            Short-term fine tune the pruned model
        3. Pick the best layer to prune
    Long-term fine tune
    """

    def __init__(self, model, config_list, trainer, evaluator, optimize_mode='maximize', experiment_data_dir='./'):
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
        experiment_data_dir : string
            PATH to save experiment data
        """
        # models used for iterative pruning and evaluation
        self._model_to_prune = copy.deepcopy(model)

        super().__init__(model, config_list)

        self._experiment_data_dir = experiment_data_dir

        # TODO: optimizer
        self._trainer = trainer
        self._evaluator = evaluator
        self._optimize_mode = OptimizeMode(optimize_mode)

        # hyper parameters for NetAdapt algorithm

        # overall pruning rate
        self._sparsity = config_list[0]['sparsity']
        # pruning rates of the layers
        self._curr_sparsity = 1
        self._delta_sparsity = 0.1

        # init current performance & best performance
        self._current_performance = -np.inf
        self._best_performance = -np.inf

        self._pruning_iteration = 0
        self._pruning_history = []

        self._config_list_level = []

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

    def _get_delta_num_weights(self):
        num_weights = []
        for wrapper in self.get_modules_wrapper():
            num_weights.append(wrapper.module.weight.data.numel())

        delta_num_weights = self._delta_sparsity * sum(num_weights)

        return delta_num_weights

    def _add_config_list(self, config_list, op_name, sparsity):
        flag = False
        for i, (sparsity, op_names) in enumerate(config_list):
            if op_name in op_names:
                self._config_list_level[i] = (
                    sparsity + sparsities[idx], op_names[i])
            if not flag:
                self._config_list_level.append(
                    {'sparsity': sparsities[idx], 'op_names': [layers[idx]]})


    def compress(self):
        """
        Compress the model.

        Returns
        -------
        torch.nn.Module
            model with specified modules compressed.
        """
        _logger.info('Starting NetAdapt Compression...')

        while self._curr_sparsity > self._sparsity:
            _logger.info('Pruning iteration: %d', self._pruning_iteration)

            # Con = Res_i - delta_Res
            self._target_sparsity = self._curr_sparsity - self._delta_sparsity
            delta_num_weights = self._get_delta_num_weights()

            # lists to store the performances of pruning different layers
            layers = []
            performances = []
            sparsities = []

            for wrapper in self.get_modules_wrapper():
                # Choose Num Filters to prune
                # TODO: check related layers
                weight_mask = wrapper.weight_mask
                sparsity = (weight_mask.numel() - weight_mask.sum().item() -
                            delta_num_weights) / weight_mask.numel()

                if sparsity <= 0:
                    _logger.info(
                        'This layer has no enough weights remained to prune')
                    continue

                # Choose which filter to prune (l1)
                config_list = [
                    {'sparsity': sparsity, 'op_names': [wrapper.name]}]
                pruner = LevelPruner(copy.deepcopy(self._model_to_prune), config_list)
                model_masked = pruner.compress()

                # Short-term fine tune the pruned model
                self.trainer(model_masked)

                perf = self.evaluator(model_masked)

                layers.append(wrapper.name)
                sparsities.append(sparsity)
                performances.append(perf)

            # 3. Pick the best layer to prune
            if self._optimize_mode is OptimizeMode.Minimize:
                idx = performances.index(min(performances))
            else:
                idx = performances.index(max(performances))

            flag = False
            for i, (sparsity, op_names) in enumerate(self._config_list_level):
                if layers[idx] in op_names:
                    self._config_list_level[i] = (
                        sparsity + sparsities[idx], op_names[i])
            if not flag:
                self._config_list_level.append(
                    {'sparsity': sparsities[idx], 'op_names': [layers[idx]]})

        _logger.info('----------Compression finished--------------')
        _logger.info('config_list generated: %s', self._config_list_level)

        # save search history
        with open(os.path.join(self._experiment_data_dir, 'search_history.csv'), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[
                'sparsity', 'performance', 'config_list'])
            writer.writeheader()
            for item in self._search_history:
                writer.writerow({'sparsity': item['sparsity'], 'performance': item['performance'], 'config_list': json.dumps(
                    item['config_list'])})

        # save best config found and best performance
        if self._optimize_mode is OptimizeMode.Minimize:
            self._best_performance *= -1
        with open(os.path.join(self._experiment_data_dir, 'search_result.json'), 'w') as jsonfile:
            json.dump({
                'performance': self._best_performance,
                'config_list': json.dumps(self._best_config_list)
            }, jsonfile)

        _logger.info('search history and result saved to foler : %s',
                     self._experiment_data_dir)

        return self.bound_model
