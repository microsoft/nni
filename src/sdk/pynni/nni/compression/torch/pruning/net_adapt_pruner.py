# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
import copy
import json
import torch
from schema import And, Optional

from nni.utils import OptimizeMode

from ..compressor import Pruner, LayerInfo
from ..utils.config_validation import CompressorSchema
from ..utils.op_dependency import get_layers_no_dependency
from .pruners import LevelPruner
from .weight_rank_filter_pruners import L1FilterPruner


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

    For the details of this algorithm, please refer to the paper: https://arxiv.org/abs/1804.03230
    """

    def __init__(self, model, config_list, evaluator, fine_tuner,
                 optimize_mode='maximize', pruning_mode='channel', pruning_step=0.05, experiment_data_dir='./'):
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
            function to evaluate the masked model
        fine_tuner : function
            function to short-term fine tune the masked model
        optimize_mode : str
            optimize mode, 'maximize' or 'minimize', by default 'maximize'
        pruning_mode : str
            'channel' or 'fine_grained', by default 'channel'
        TODO: refine
        pruning_step : float
            sparsity to prune in each iteration
        experiment_data_dir : str
            PATH to save experiment data
        """
        # models used for iterative pruning and evaluation
        self._model_to_prune = copy.deepcopy(model)
        self._pruning_mode = pruning_mode

        super().__init__(model, config_list)

        self._fine_tuner = fine_tuner
        self._evaluator = evaluator
        self._optimize_mode = OptimizeMode(optimize_mode)

        # hyper parameters for NetAdapt algorithm
        self._pruning_step = pruning_step

        # overall pruning rate
        self._sparsity = config_list[0]['sparsity']

        # config_list
        self._config_list = []

        self._experiment_data_dir = experiment_data_dir
        if not os.path.exists(self._experiment_data_dir):
            os.makedirs(self._experiment_data_dir)

        self._tmp_model_path = os.path.join(
            self._experiment_data_dir, 'tmp_model.pth')

        self._total_num_weights = self._get_total_num_weights()

    def _detect_modules_to_compress(self):
        """
        redefine this function, consider only the layers without dependencies
        """
        self._op_names_to_compress = []
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
                    self._op_names_to_compress.append(name)
        return self.modules_to_compress

    def _get_total_num_weights(self, count_non_prunable_ops=True):
        '''
        calculate the total number of weights

        Parameters
        ----------
        count_non_prunable_ops : bool
            indicate if non prunable ops should be considerd

        Returns
        -------
        int
            total weights of all the op considered
        '''
        num_weights = 0
        if count_non_prunable_ops:
            pruner = LevelPruner(
                copy.deepcopy(self._model_to_prune), [{'sparsity': 0.1, 'op_types': ['default']}])
            modules_wrapper = pruner.get_modules_wrapper()
        else:
            modules_wrapper = self.get_modules_wrapper()

        for wrapper in modules_wrapper:
            _logger.debug("num_weights of op %s: %d",
                          wrapper.name, wrapper.module.weight.data.numel())
            num_weights += wrapper.module.weight.data.numel()
        _logger.info("Total num weights: %d", num_weights)

        return num_weights

    def _get_delta_num_weights(self):
        delta_num_weights = self._pruning_step * self._total_num_weights

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

    def _get_op_num_weights_remained(self, op_name):
        '''
        Get the number of weights remained after channel pruning with current sparsity

        Returns
        -------
        int
            remained number of weights of the op
        '''
        for wrapper in self.get_modules_wrapper():
            if wrapper.name == op_name:
                return wrapper.weight_mask.sum().item()

    def _get_op_sparsity(self, op_name):
        for config in self._config_list:
            if 'op_names' in config and op_name in config['op_names']:
                return config['sparsity']
        return 0

    def _calc_related_weights(self, op_name):
        '''
        Calculate total number weights of the op and the next op, applicable only for models without dependencies among ops

        Parameters
        ----------
        op_name : str

        Returns
        -------
        int
            total number of all the realted (current and the next) op weights
        '''
        num_weights = 0
        flag_found = False
        previous_name = None
        previous_module = None

        for name, module in self._model_to_prune.named_modules():
            if module == self.bound_model:
                continue
            if not flag_found and name == op_name:
                _logger.debug("original module found: %s", name)
                num_weights = module.weight.data.numel()

                # consider filter pruned in this op caused by previous op's channel pruning
                if previous_module and type(module).__name__ in ['Conv2d'] and type(previous_module).__name__ in ['Conv2d']:
                    sparsity_previous_op = self._get_op_sparsity(previous_name)
                    if sparsity_previous_op:
                        _logger.debug(
                            "decrease op's weights by %s due to previous op %s's channel pruning...", sparsity_previous_op, name)
                        num_weights *= (1-sparsity_previous_op)

                flag_found = True
                continue
            # TODO: check Functional ops, dropout, activation...
            # skip Dropout2d
            if flag_found and type(module).__name__ in ['Dropout2d']:
                continue
            if flag_found and type(module).__name__ in ['Conv2d', 'Linear']:
                _logger.debug("related module found: %s", name)
                # channel/filter pruning crossing is considered here, so only the num_weights after channel pruning is valuable
                # TODO: fine-grained not supported
                if name in self._op_names_to_compress:
                    num_weights += self._get_op_num_weights_remained(name)
                else:
                    num_weights += module.weight.data.numel()
                break
            previous_name = name
            previous_module = module

        return num_weights

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
            target_sparsity = current_sparsity + self._pruning_step
            delta_num_weights = self._get_delta_num_weights()

            # variable to store the info of the best layer found in this iteration
            best_layer = {}

            for wrapper in self.get_modules_wrapper():
                _logger.debug("op name : %s", wrapper.name)
                _logger.debug("op weights : %d", wrapper.weight_mask.numel())
                _logger.debug("op left weights : %d",
                              wrapper.weight_mask.sum().item())
                _logger.debug("related weights : %d",
                              self._calc_related_weights(wrapper.name))

                current_op_sparsity = 1 - wrapper.weight_mask.sum().item() / \
                    wrapper.weight_mask.numel()
                _logger.debug("current op sparsity : %s", current_op_sparsity)

                # sparsity that this layer needs to prune to satisfy the requirement
                target_op_sparsity = current_op_sparsity + \
                    delta_num_weights / \
                    self._calc_related_weights(wrapper.name)

                # TODO: round the target_op_sparsity to real op_sparsity base on the op shape
                if target_op_sparsity >= 1:
                    _logger.info(
                        'Layer %s has no enough weights (remained) to prune', wrapper.name)
                    continue

                config_list = self._update_config_list(
                    self._config_list, wrapper.name, target_op_sparsity)
                _logger.info("config_list used : %s", config_list)

                if self._pruning_mode == 'channel':
                    pruner = L1FilterPruner(copy.deepcopy(
                        self._model_to_prune), config_list)
                elif self._pruning_mode == 'fine_grained':
                    pruner = LevelPruner(copy.deepcopy(
                        self._model_to_prune), config_list)
                model_masked = pruner.compress()

                performance = self._evaluator(model_masked)
                _logger.info(
                    "Layer : %s, evaluation result before fine tuning : %s", wrapper.name, performance)
                # Short-term fine tune the pruned model
                self._fine_tuner(model_masked)

                performance = self._evaluator(model_masked)
                _logger.info(
                    "Layer : %s, evaluation result after short-term fine tuning : %s", wrapper.name, performance)

                if not best_layer \
                    or (self._optimize_mode is OptimizeMode.Maximize and performance > best_layer['performance']) \
                    or (self._optimize_mode is OptimizeMode.Minimize and performance < best_layer['performance']):
                    _logger.debug("updating best layer to %s...", wrapper.name)
                    # find weight mask of this layer
                    for w in pruner.get_modules_wrapper():
                        if w.name == wrapper.name:
                            masks = {'weight_mask': w.weight_mask,
                                     'bias_mask': w.bias_mask}
                            break
                    best_layer = {
                        'op_name': wrapper.name,
                        'sparsity': target_op_sparsity,
                        'performance': performance,
                        'masks': masks
                    }

                    # save model weights
                    pruner.export_model(self._tmp_model_path)

            if not best_layer:
                self._pruning_step *= 0.5
                _logger.info("No more layers to prune, decrease pruning step to %s", self._pruning_step)
                continue

            # Pick the best layer to prune, update iterative information
            # update config_list
            self._config_list = self._update_config_list(
                self._config_list, best_layer['op_name'], best_layer['sparsity'])

            # update weights parameters
            self._model_to_prune.load_state_dict(
                torch.load(self._tmp_model_path))

            # update mask of the chosen op
            for wrapper in self.get_modules_wrapper():
                if wrapper.name == best_layer['op_name']:
                    for k in masks:
                        setattr(wrapper, k, masks[k])
                    break

            current_sparsity = target_sparsity
            _logger.info('Pruning iteration %d finished.', pruning_iteration)
            _logger.info('Layer %s seleted with sparsity %s, performance after pruning & short term fine-tuning : %s, \
                         current overall sparsity : %s',
                         best_layer['op_name'], best_layer['sparsity'], best_layer['performance'], current_sparsity)
            pruning_iteration += 1

            self._final_performance = best_layer['performance']

        # load weights parameters
        self.load_model_state_dict(torch.load(self._tmp_model_path))
        os.remove(self._tmp_model_path)

        _logger.info('----------Compression finished--------------')
        _logger.info('config_list generated: %s', self._config_list)
        _logger.info("Performance after pruning: %s", self._final_performance)
        _logger.info("Masked sparsity: %.6f", current_sparsity)

        # save best config found and best performance
        with open(os.path.join(self._experiment_data_dir, 'search_result.json'), 'w') as jsonfile:
            json.dump({
                'performance': self._final_performance,
                'config_list': json.dumps(self._config_list)
            }, jsonfile)

        _logger.info('search history and result saved to foler : %s',
                     self._experiment_data_dir)

        return self.bound_model
