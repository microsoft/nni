# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Optional, Callable

import json_tricks
from torch.nn import Module
from torch.optim import Optimizer

import nni
from .constants import PRUNER_DICT, QUANTIZER_DICT
from .interface import BaseAutoCompressionEngine, AbstractAutoCompressionModule
from .utils import import_

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

class AutoCompressionEngine(BaseAutoCompressionEngine):
    @classmethod
    def __convert_compact_pruner_params_to_config_list(cls, compact_config: dict) -> list:
        config_dict = {}
        for key, value in compact_config.items():
            _, op_types, op_names, var_name = key.split('::')
            config_dict.setdefault((op_types, op_names), {})
            config_dict[(op_types, op_names)][var_name] = value

        config_list = []
        for key, config in config_dict.items():
            op_types, op_names = key
            op_types = op_types.split(':') if op_types else []
            op_names = op_names.split(':') if op_names else []
            if op_types:
                config['op_types'] = op_types
            if op_names:
                config['op_names'] = op_names
            if 'op_types' in config or 'op_names' in config:
                config_list.append(config)

        return config_list

    @classmethod
    def __convert_compact_quantizer_params_to_config_list(cls, compact_config: dict) -> list:
        config_dict = {}
        for key, value in compact_config.items():
            _, quant_types, op_types, op_names, var_name = key.split('::')
            config_dict.setdefault((quant_types, op_types, op_names), {})
            config_dict[(quant_types, op_types, op_names)][var_name] = value

        config_list = []
        for key, config in config_dict.items():
            quant_types, op_types, op_names = key
            quant_types = quant_types.split(':')
            op_types = op_types.split(':')
            op_names = op_names.split(':')
            if quant_types:
                config['quant_types'] = quant_types
            else:
                continue
            if op_types:
                config['op_types'] = op_types
            if op_names:
                config['op_names'] = op_names
            if 'op_types' in config or 'op_names' in config:
                config_list.append(config)

        return config_list

    @classmethod
    def _convert_compact_params_to_config_list(cls, compressor_type: str, compact_config: dict) -> list:
        func_dict = {
            'pruner': cls.__convert_compact_pruner_params_to_config_list,
            'quantizer': cls.__convert_compact_quantizer_params_to_config_list
        }
        return func_dict[compressor_type](compact_config)

    @classmethod
    def __compress_pruning(cls, algorithm_name: str,
                           model: Module,
                           config_list: list,
                           optimizer_factory: Optional[Callable],
                           criterion: Optional[Callable],
                           sparsifying_trainer: Optional[Callable[[Module, Optimizer, Callable, int], None]],
                           finetuning_trainer: Optional[Callable[[Module, Optimizer, Callable, int], None]],
                           finetuning_epochs: int,
                           **compressor_parameter_dict) -> Module:
        if algorithm_name in ['level', 'l1', 'l2', 'fpgm']:
            pruner = PRUNER_DICT[algorithm_name](model, config_list, **compressor_parameter_dict)
        elif algorithm_name in ['slim', 'taylorfo', 'apoz', 'mean_activation']:
            optimizer = None if optimizer_factory is None else optimizer_factory(model.parameters())
            pruner = PRUNER_DICT[algorithm_name](model, config_list, optimizer, sparsifying_trainer, criterion, **compressor_parameter_dict)
        else:
            raise ValueError('Unsupported compression algorithm: {}.'.format(algorithm_name))
        compressed_model = pruner.compress()
        if finetuning_trainer is not None:
            # note that in pruning process, finetuning will use an un-patched optimizer
            optimizer = optimizer_factory(compressed_model.parameters())
            for i in range(finetuning_epochs):
                finetuning_trainer(compressed_model, optimizer, criterion, i)
        pruner.get_pruned_weights()
        return compressed_model

    @classmethod
    def __compress_quantization(cls, algorithm_name: str,
                                model: Module,
                                config_list: list,
                                optimizer_factory: Optional[Callable],
                                criterion: Optional[Callable],
                                sparsifying_trainer: Optional[Callable[[Module, Optimizer, Callable, int], None]],
                                finetuning_trainer: Optional[Callable[[Module, Optimizer, Callable, int], None]],
                                finetuning_epochs: int,
                                **compressor_parameter_dict) -> Module:
        optimizer = None if optimizer_factory is None else optimizer_factory(model.parameters())
        quantizer = QUANTIZER_DICT[algorithm_name](model, config_list, optimizer, **compressor_parameter_dict)
        compressed_model = quantizer.compress()
        if finetuning_trainer is not None:
            # note that in quantization process, finetuning will use a patched optimizer
            for i in range(finetuning_epochs):
                finetuning_trainer(compressed_model, optimizer, criterion, i)
        return compressed_model

    @classmethod
    def _compress(cls, compressor_type: str,
                  algorithm_name: str,
                  model: Module,
                  config_list: list,
                  optimizer_factory: Optional[Callable],
                  criterion: Optional[Callable],
                  sparsifying_trainer: Optional[Callable[[Module, Optimizer, Callable, int], None]],
                  finetuning_trainer: Optional[Callable[[Module, Optimizer, Callable, int], None]],
                  finetuning_epochs: int,
                  **compressor_parameter_dict) -> Module:
        func_dict = {
            'pruner': cls.__compress_pruning,
            'quantizer': cls.__compress_quantization
        }
        _logger.info('%s compressor config_list:\n%s', algorithm_name, json_tricks.dumps(config_list, indent=4))
        compressed_model = func_dict[compressor_type](algorithm_name, model, config_list, optimizer_factory, criterion, sparsifying_trainer,
                                                      finetuning_trainer, finetuning_epochs, **compressor_parameter_dict)
        return compressed_model

    @classmethod
    def trial_execute_compress(cls, module_name):
        auto_compress_module: AbstractAutoCompressionModule = import_(module_name)

        algorithm_config = nni.get_next_parameter()['algorithm_name']
        algorithm_name = algorithm_config['_name']
        compact_config = {k: v for k, v in algorithm_config.items() if k.startswith('config_list::')}
        parameter_dict = {k.split('parameter::')[1]: v for k, v in algorithm_config.items() if k.startswith('parameter::')}

        compressor_type = 'quantizer' if algorithm_name in QUANTIZER_DICT else 'pruner'

        config_list = cls._convert_compact_params_to_config_list(compressor_type, compact_config)

        model, evaluator = auto_compress_module.model(), auto_compress_module.evaluator()
        optimizer_factory, criterion = auto_compress_module.optimizer_factory(), auto_compress_module.criterion()
        sparsifying_trainer = auto_compress_module.sparsifying_trainer(algorithm_name)
        finetuning_trainer = auto_compress_module.post_compress_finetuning_trainer(algorithm_name)
        finetuning_epochs = auto_compress_module.post_compress_finetuning_epochs(algorithm_name)

        compressed_model = cls._compress(compressor_type, algorithm_name, model, config_list, optimizer_factory,
                                         criterion, sparsifying_trainer, finetuning_trainer, finetuning_epochs, **parameter_dict)

        nni.report_final_result(evaluator(compressed_model))
