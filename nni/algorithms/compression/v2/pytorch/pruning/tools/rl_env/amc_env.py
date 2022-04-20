# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import OrderedDict
from copy import Error
import logging
from typing import Dict, List

import numpy as np
from torch import Tensor
from torch.nn import Module

from nni.algorithms.compression.v2.pytorch.utils import config_list_canonical
from nni.compression.pytorch.utils.counter import count_flops_params

_logger = logging.getLogger(__name__)


class AMCEnv:
    def __init__(self, model: Module, config_list: List[Dict], dummy_input: Tensor, total_sparsity: float, max_sparsity_per_layer: Dict[str, float], target: str = 'flops'):
        pruning_op_names = []
        [pruning_op_names.extend(config['op_names']) for config in config_list_canonical(model, config_list)]
        self.pruning_ops = OrderedDict()
        self.pruning_types = []
        for i, (name, layer) in enumerate(model.named_modules()):
            if name in pruning_op_names:
                op_type = type(layer).__name__
                stride = np.power(np.prod(layer.stride), 1 / len(layer.stride)) if hasattr(layer, 'stride') else 0
                kernel_size = np.power(np.prod(layer.kernel_size), 1 / len(layer.kernel_size)) if hasattr(layer, 'kernel_size') else 1
                self.pruning_ops[name] = (i, op_type, stride, kernel_size)
                self.pruning_types.append(op_type)
        self.pruning_types = list(set(self.pruning_types))
        self.pruning_op_names = list(self.pruning_ops.keys())
        self.dummy_input = dummy_input

        self.total_sparsity = total_sparsity
        self.max_sparsity_per_layer = max_sparsity_per_layer
        assert target in ['flops', 'params']
        self.target = target

        self.origin_target, self.origin_params_num, self.origin_statistics = count_flops_params(model, dummy_input, verbose=False)
        self.origin_statistics = {result['name']: result for result in self.origin_statistics}

        self.under_pruning_target = sum([self.origin_statistics[name][self.target] for name in self.pruning_op_names])
        self.excepted_pruning_target = self.total_sparsity * self.under_pruning_target

    def reset(self):
        self.ops_iter = iter(self.pruning_ops)
        # build embedding (static part)
        self._build_state_embedding(self.origin_statistics)
        observation = self.layer_embedding[0].copy()
        return observation

    def correct_action(self, action: float, model: Module):
        try:
            op_name = next(self.ops_iter)
            index = self.pruning_op_names.index(op_name)
            _, _, current_statistics = count_flops_params(model, self.dummy_input, verbose=False)
            current_statistics = {result['name']: result for result in current_statistics}

            total_current_target = sum([current_statistics[name][self.target] for name in self.pruning_op_names])
            previous_pruning_target = self.under_pruning_target - total_current_target
            max_rest_pruning_target = sum([current_statistics[name][self.target] * self.max_sparsity_per_layer[name] for name in self.pruning_op_names[index + 1:]])
            min_current_pruning_target = self.excepted_pruning_target - previous_pruning_target - max_rest_pruning_target
            max_current_pruning_target_1 = self.origin_statistics[op_name][self.target] * self.max_sparsity_per_layer[op_name] - (self.origin_statistics[op_name][self.target] - current_statistics[op_name][self.target])
            max_current_pruning_target_2 = self.excepted_pruning_target - previous_pruning_target
            max_current_pruning_target = min(max_current_pruning_target_1, max_current_pruning_target_2)
            min_action = min_current_pruning_target / current_statistics[op_name][self.target]
            max_action = max_current_pruning_target / current_statistics[op_name][self.target]
            if min_action > self.max_sparsity_per_layer[op_name]:
                _logger.warning('[%s] min action > max sparsity per layer: %f > %f', op_name, min_action, self.max_sparsity_per_layer[op_name])
            action = max(0., min(max_action, max(min_action, action)))

            self.current_op_name = op_name
            self.current_op_target = current_statistics[op_name][self.target]
        except StopIteration:
            raise Error('Something goes wrong, this should not happen.')
        return action

    def step(self, action: float, model: Module):
        _, _, current_statistics = count_flops_params(model, self.dummy_input, verbose=False)
        current_statistics = {result['name']: result for result in current_statistics}
        index = self.pruning_op_names.index(self.current_op_name)
        action = 1 - current_statistics[self.current_op_name][self.target] / self.current_op_target

        total_current_target = sum([current_statistics[name][self.target] for name in self.pruning_op_names])
        previous_pruning_target = self.under_pruning_target - total_current_target
        rest_target = sum([current_statistics[name][self.target] for name in self.pruning_op_names[index + 1:]])

        self.layer_embedding[index][-3] = previous_pruning_target / self.under_pruning_target  # reduced
        self.layer_embedding[index][-2] = rest_target / self.under_pruning_target  # rest
        self.layer_embedding[index][-1] = action  # last action
        observation = self.layer_embedding[index, :].copy()

        return action, 0, observation, self.is_final_layer()

    def is_first_layer(self):
        return self.pruning_op_names.index(self.current_op_name) == 0

    def is_final_layer(self):
        return self.pruning_op_names.index(self.current_op_name) == len(self.pruning_op_names) - 1

    @property
    def state_feature(self):
        return ['index', 'layer_type', 'input_size', 'output_size', 'stride', 'kernel_size', 'params_size', 'reduced', 'rest', 'a_{t-1}']

    def _build_state_embedding(self, statistics: Dict[str, Dict]):
        _logger.info('Building state embedding...')
        layer_embedding = []
        for name, (idx, op_type, stride, kernel_size) in self.pruning_ops.items():
            state = []
            state.append(idx)  # index
            state.append(self.pruning_types.index(op_type))  # layer type
            state.append(np.prod(statistics[name]['input_size']))  # input size
            state.append(np.prod(statistics[name]['output_size']))  # output size
            state.append(stride)  # stride
            state.append(kernel_size)  # kernel size
            state.append(statistics[name]['params'])  # params size
            state.append(0.)  # reduced
            state.append(1.)  # rest
            state.append(0.)  # a_{t-1}
            layer_embedding.append(np.array(state))
        layer_embedding = np.array(layer_embedding, 'float')
        _logger.info('=> shape of embedding (n_layer * n_dim): %s', layer_embedding.shape)
        assert len(layer_embedding.shape) == 2, layer_embedding.shape

        # normalize the state
        for i in range(layer_embedding.shape[1]):
            fmin = min(layer_embedding[:, i])
            fmax = max(layer_embedding[:, i])
            if fmax - fmin > 0:
                layer_embedding[:, i] = (layer_embedding[:, i] - fmin) / (fmax - fmin)

        self.layer_embedding = layer_embedding
