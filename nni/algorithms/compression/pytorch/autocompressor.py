# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import logging
import os
import time
import torch

from collections import OrderedDict
from pathlib import Path
from typing import Tuple, Union, List, Optional

from nni.algorithms.compression.pytorch.multicompressor import MultiCompressor

_logger = logging.getLogger(__name__)


LAYER_PRUNER_MAPPING = {
    'Conv2d': ['l1', 'l2'],
    'Linear': ['level'],
    'BatchNorm2d': ['slim']
}

LAYER_QUANTIZER_MAPPING = {
    'Conv2d': ['qat'],
    'Linear': ['qat']
}

class AutoCompressor:
    def __init__(self, model, sparsity: float = 0.5, optimizer=None, trainer=None):
        self.bound_model = model
        self.sparsity = sparsity
        self.optimizer = optimizer
        self.trainer = trainer
        self.pruner_layer_dict, self.quantizer_layer_dict, self.max_count = self._detect_search_space()
        self.counter = 0

    def _detect_search_space(self) -> Tuple[OrderedDict, OrderedDict, int]:
        pruner_layer = set()
        quantizer_layer = set()
        for _, module in self.bound_model.named_modules():
            layer_type = type(module).__name__
            if module == self.bound_model:
                continue
            if layer_type in LAYER_PRUNER_MAPPING:
                pruner_layer.add(layer_type)
            else:
                _logger.debug('Unsupported auto pruning layer: %s', layer_type)
            if layer_type in LAYER_QUANTIZER_MAPPING:
                quantizer_layer.add(layer_type)
            else:
                _logger.debug('Unsupported auto quantizing layer: %s', layer_type)

        assert len(pruner_layer) + len(quantizer_layer) > 0, 'The model has no supported layer to compress.'

        total_combination_num = 1

        pruner_layer_dict = OrderedDict()
        for layer_name in pruner_layer:
            pruner_layer_dict.setdefault(layer_name, [])
            for pruner_type in LAYER_PRUNER_MAPPING[layer_name]:
                pruner_layer_dict[layer_name].append((pruner_type, {}, {'sparsity': self.sparsity, 'op_types': [layer_name]}))
            total_combination_num *= len(pruner_layer_dict[layer_name])

        quantizer_layer_dict = OrderedDict()
        for layer_name in quantizer_layer:
            quantizer_layer_dict.setdefault(layer_name, [])
            for quantizer_type in LAYER_QUANTIZER_MAPPING[layer_name]:
                quantizer_layer_dict[layer_name].append((quantizer_type, {}, {'quant_types': ['weight'], 'quant_bits': {'weight': 8}, 'op_types': [layer_name]}))
            total_combination_num *= len(quantizer_layer_dict[layer_name])

        return pruner_layer_dict, quantizer_layer_dict, int(total_combination_num)

    def _generate_config_list(self):
        quo, rem = self.counter, 0
        pruner_config_dict = {}
        for layer_name, choices in self.pruner_layer_dict.items():
            quo, rem = quo // len(choices), quo % len(choices)
            name, args, config = choices[rem]
            pruner_config_dict.setdefault(name, {'config_list': [], 'pruner': {'type': name, 'args': args}})
            pruner_config_dict[name]['config_list'].append(config)

        quantizer_config_dict = {}
        for layer_name, choices in self.quantizer_layer_dict.items():
            quo, rem = quo // len(choices), quo % len(choices)
            name, args, config = choices[rem]
            quantizer_config_dict.setdefault(name, {'config_list': [], 'quantizer': {'type': name, 'args': args}})
            quantizer_config_dict[name]['config_list'].append(config)

        self.counter += 1

        return list(pruner_config_dict.values()) + list(quantizer_config_dict.values())

    def run(self, result_dir: str = None, input_shape: Optional[Union[List, Tuple]] = None, device: torch.device = None):
        if result_dir is None:
            result_dir = os.path.abspath('./autocompress_result_{}'.format(int(time.time())))
        while self.counter < self.max_count:
            mixed_config_list = self._generate_config_list()
            config_result_dir = os.path.join(result_dir, str(self.counter))
            Path(config_result_dir).mkdir(parents=True, exist_ok=True)
            _logger.info('Result saved under %s', config_result_dir)

            compressor = MultiCompressor(copy.deepcopy(self.bound_model), mixed_config_list, optimizer=copy.deepcopy(self.optimizer), trainer=self.trainer)
            compressor.set_config(os.path.join(config_result_dir, 'model.pt'), os.path.join(config_result_dir, 'mask.pt'),
                                  os.path.join(config_result_dir, 'calibration.pt'), os.path.join(config_result_dir, 'onnx.pt'),
                                  input_shape=input_shape, device=device)
            compressor.compress()
