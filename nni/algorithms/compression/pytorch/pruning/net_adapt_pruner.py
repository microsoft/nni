# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
import copy
import json
import torch
from schema import And, Optional

from nni.utils import OptimizeMode

from nni.compression.pytorch.compressor import Pruner
from nni.compression.pytorch.utils.config_validation import CompressorSchema
from nni.compression.pytorch.utils.num_param_counter import get_total_num_weights
from .constants_pruner import PRUNER_DICT


_logger = logging.getLogger(__name__)


class NetAdaptPruner(Pruner):
    """
    A Pytorch implementation of NetAdapt compression algorithm.

    Parameters
    ----------
    model : pytorch model
        The model to be pruned.
    config_list : list
        Supported keys:
            - sparsity : The target overall sparsity.
            - op_types : The operation type to prune.
    short_term_fine_tuner : function
        function to short-term fine tune the masked model.
        This function should include `model` as the only parameter,
        and fine tune the model for a short term after each pruning iteration.
        Example::

            def short_term_fine_tuner(model, epoch=3):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                train_loader = ...
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                model.train()
                for _ in range(epoch):
                    for batch_idx, (data, target) in enumerate(train_loader):
                        data, target = data.to(device), target.to(device)
                        optimizer.zero_grad()
                        output = model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()
    evaluator : function
        function to evaluate the masked model.
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
        optimize mode, `maximize` or `minimize`, by default `maximize`.
    base_algo : str
        Base pruning algorithm. `level`, `l1`, `l2` or `fpgm`, by default `l1`. Given the sparsity distribution among the ops,
        the assigned `base_algo` is used to decide which filters/channels/weights to prune.
    sparsity_per_iteration : float
        sparsity to prune in each iteration.
    experiment_data_dir : str
        PATH to save experiment data,
        including the config_list generated for the base pruning algorithm and the performance of the pruned model.
    """

    def __init__(self, model, config_list, short_term_fine_tuner, evaluator,
                 optimize_mode='maximize', base_algo='l1', sparsity_per_iteration=0.05, experiment_data_dir='./'):
        # models used for iterative pruning and evaluation
        self._model_to_prune = copy.deepcopy(model)
        self._base_algo = base_algo

        super().__init__(model, config_list)

        self._short_term_fine_tuner = short_term_fine_tuner
        self._evaluator = evaluator
        self._optimize_mode = OptimizeMode(optimize_mode)

        # hyper parameters for NetAdapt algorithm
        self._sparsity_per_iteration = sparsity_per_iteration

        # overall pruning rate
        self._sparsity = config_list[0]['sparsity']

        # config_list
        self._config_list_generated = []

        self._experiment_data_dir = experiment_data_dir
        if not os.path.exists(self._experiment_data_dir):
            os.makedirs(self._experiment_data_dir)

        self._tmp_model_path = os.path.join(self._experiment_data_dir, 'tmp_model.pth')

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
        elif self._base_algo in ['l1', 'l2', 'fpgm']:
            schema = CompressorSchema([{
                'sparsity': And(float, lambda n: 0 < n < 1),
                'op_types': ['Conv2d'],
                Optional('op_names'): [str]
            }], model, _logger)

        schema.validate(config_list)

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

        # if op_name is not in self._config_list_generated, create a new json item
        if self._base_algo in ['l1', 'l2', 'fpgm']:
            config_list_updated.append(
                {'sparsity': sparsity, 'op_types': ['Conv2d'], 'op_names': [op_name]})
        elif self._base_algo == 'level':
            config_list_updated.append(
                {'sparsity': sparsity, 'op_names': [op_name]})

        return config_list_updated

    def _get_op_num_weights_remained(self, op_name, module):
        '''
        Get the number of weights remained after channel pruning with current sparsity

        Returns
        -------
        int
            remained number of weights of the op
        '''

        # if op is wrapped by the pruner
        for wrapper in self.get_modules_wrapper():
            if wrapper.name == op_name:
                return wrapper.weight_mask.sum().item()

        # if op is not wrapped by the pruner
        return module.weight.data.numel()

    def _get_op_sparsity(self, op_name):
        for config in self._config_list_generated:
            if 'op_names' in config and op_name in config['op_names']:
                return config['sparsity']
        return 0

    def _calc_num_related_weights(self, op_name):
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
            if not flag_found and name != op_name and type(module).__name__ in ['Conv2d', 'Linear']:
                previous_name = name
                previous_module = module
            if not flag_found and name == op_name:
                _logger.debug("original module found: %s", name)
                num_weights = module.weight.data.numel()

                # consider related pruning in this op caused by previous op's pruning
                if previous_module:
                    sparsity_previous_op = self._get_op_sparsity(previous_name)
                    if sparsity_previous_op:
                        _logger.debug(
                            "decrease op's weights by %s due to previous op %s's pruning...", sparsity_previous_op, previous_name)
                        num_weights *= (1-sparsity_previous_op)

                flag_found = True
                continue
            if flag_found and type(module).__name__ in ['Conv2d', 'Linear']:
                _logger.debug("related module found: %s", name)
                # channel/filter pruning crossing is considered here, so only the num_weights after channel pruning is valuable
                num_weights += self._get_op_num_weights_remained(name, module)
                break

        _logger.debug("num related weights of op %s : %d", op_name, num_weights)

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
        delta_num_weights_per_iteration = \
            int(get_total_num_weights(self._model_to_prune, ['Conv2d', 'Linear']) * self._sparsity_per_iteration)

        # stop condition
        while current_sparsity < self._sparsity:
            _logger.info('Pruning iteration: %d', pruning_iteration)

            # calculate target sparsity of this iteration
            target_sparsity = current_sparsity + self._sparsity_per_iteration

            # variable to store the info of the best layer found in this iteration
            best_op = {}

            for wrapper in self.get_modules_wrapper():
                _logger.debug("op name : %s", wrapper.name)
                _logger.debug("op weights : %d", wrapper.weight_mask.numel())
                _logger.debug("op left weights : %d", wrapper.weight_mask.sum().item())

                current_op_sparsity = 1 - wrapper.weight_mask.sum().item() / wrapper.weight_mask.numel()
                _logger.debug("current op sparsity : %s", current_op_sparsity)

                # sparsity that this layer needs to prune to satisfy the requirement
                target_op_sparsity = current_op_sparsity + delta_num_weights_per_iteration / self._calc_num_related_weights(wrapper.name)

                if target_op_sparsity >= 1:
                    _logger.info('Layer %s has no enough weights (remained) to prune', wrapper.name)
                    continue

                config_list = self._update_config_list(self._config_list_generated, wrapper.name, target_op_sparsity)
                _logger.debug("config_list used : %s", config_list)

                pruner = PRUNER_DICT[self._base_algo](copy.deepcopy(self._model_to_prune), config_list)
                model_masked = pruner.compress()

                # Short-term fine tune the pruned model
                self._short_term_fine_tuner(model_masked)

                performance = self._evaluator(model_masked)
                _logger.info("Layer : %s, evaluation result after short-term fine tuning : %s", wrapper.name, performance)

                if not best_op \
                    or (self._optimize_mode is OptimizeMode.Maximize and performance > best_op['performance']) \
                    or (self._optimize_mode is OptimizeMode.Minimize and performance < best_op['performance']):
                    _logger.debug("updating best layer to %s...", wrapper.name)
                    # find weight mask of this layer
                    for w in pruner.get_modules_wrapper():
                        if w.name == wrapper.name:
                            masks = {'weight_mask': w.weight_mask,
                                     'bias_mask': w.bias_mask}
                            break
                    best_op = {
                        'op_name': wrapper.name,
                        'sparsity': target_op_sparsity,
                        'performance': performance,
                        'masks': masks
                    }

                    # save model weights
                    pruner.export_model(self._tmp_model_path)

            if not best_op:
                # decrease pruning step
                self._sparsity_per_iteration *= 0.5
                _logger.info("No more layers to prune, decrease pruning step to %s", self._sparsity_per_iteration)
                continue

            # Pick the best layer to prune, update iterative information
            # update config_list
            self._config_list_generated = self._update_config_list(
                self._config_list_generated, best_op['op_name'], best_op['sparsity'])

            # update weights parameters
            self._model_to_prune.load_state_dict(torch.load(self._tmp_model_path))

            # update mask of the chosen op
            for wrapper in self.get_modules_wrapper():
                if wrapper.name == best_op['op_name']:
                    for k in best_op['masks']:
                        setattr(wrapper, k, best_op['masks'][k])
                    break

            current_sparsity = target_sparsity
            _logger.info('Pruning iteration %d finished, current sparsity: %s', pruning_iteration, current_sparsity)
            _logger.info('Layer %s seleted with sparsity %s, performance after pruning & short term fine-tuning : %s',
                         best_op['op_name'], best_op['sparsity'], best_op['performance'])
            pruning_iteration += 1

            self._final_performance = best_op['performance']

        # load weights parameters
        self.load_model_state_dict(torch.load(self._tmp_model_path))
        os.remove(self._tmp_model_path)

        _logger.info('----------Compression finished--------------')
        _logger.info('config_list generated: %s', self._config_list_generated)
        _logger.info("Performance after pruning: %s", self._final_performance)
        _logger.info("Masked sparsity: %.6f", current_sparsity)

        # save best config found and best performance
        with open(os.path.join(self._experiment_data_dir, 'search_result.json'), 'w') as jsonfile:
            json.dump({
                'performance': self._final_performance,
                'config_list': json.dumps(self._config_list_generated)
            }, jsonfile)

        _logger.info('search history and result saved to foler : %s', self._experiment_data_dir)

        return self.bound_model
