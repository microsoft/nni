# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Optional, Callable

import json_tricks
from torch.nn import Module
from torch.optim import Optimizer

import nni
from .constants import PRUNER_DICT, QUANTIZER_DICT
from .interface import AbstractExecutionEngine
from .utils import import_

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

class AutoCompressEngine(AbstractExecutionEngine):
    @classmethod
    def __convert_pruner_config_list(cls, converted_config_dict: dict) -> list:
        config_dict = {}
        for key, value in converted_config_dict.items():
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
    def __convert_quantizer_config_list(cls, converted_config_dict: dict) -> list:
        config_dict = {}
        for key, value in converted_config_dict.items():
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
    def _convert_config_list(cls, compressor_type: str, converted_config_dict: dict) -> list:
        func_dict = {
            'pruner': cls.__convert_pruner_config_list,
            'quantizer': cls.__convert_quantizer_config_list
        }
        return func_dict[compressor_type](converted_config_dict)

    @classmethod
    def __compress_pruning_pipeline(cls, algorithm_name: str,
                                    model: Module,
                                    config_list: list,
                                    evaluator: Callable[[Module], float],
                                    optimizer: Optional[Optimizer],
                                    trainer: Optional[Callable[[Module, Optimizer], None]],
                                    finetune_trainer: Optional[Callable[[Module, Optimizer], None]],
                                    **compressor_parameter_dict) -> Module:
        # evaluator is for future use
        pruner = PRUNER_DICT[algorithm_name](model, config_list, optimizer, **compressor_parameter_dict)
        model = pruner.compress()
        if trainer:
            trainer(model)
        if finetune_trainer:
            finetune_trainer(model)
        return model

    @classmethod
    def __compress_quantization_pipeline(cls, algorithm_name: str,
                                         model: Module,
                                         config_list: list,
                                         evaluator: Callable[[Module], float],
                                         optimizer: Optional[Optimizer],
                                         trainer: Callable[[Module, Optimizer], None],
                                         finetune_trainer: Optional[Callable[[Module, Optimizer], None]],
                                         **compressor_parameter_dict) -> Module:
        # evaluator is for future use
        quantizer = QUANTIZER_DICT[algorithm_name](model, config_list, optimizer, **compressor_parameter_dict)
        model = quantizer.compress()
        if trainer:
            trainer(model)
        if finetune_trainer:
            finetune_trainer(model)
        return model

    @classmethod
    def _compress_pipeline(cls, compressor_type: str,
                           algorithm_name: str,
                           model: Module,
                           config_list: list,
                           evaluator: Callable[[Module], float],
                           optimizer: Optional[Optimizer],
                           trainer: Optional[Callable[[Module, Optimizer], None]],
                           finetune_trainer: Optional[Callable[[Module, Optimizer], None]],
                           **compressor_parameter_dict) -> Module:
        func_dict = {
            'pruner': cls.__compress_pruning_pipeline,
            'quantizer': cls.__compress_quantization_pipeline
        }
        _logger.info('%s compressor config_list:\n%s', algorithm_name, json_tricks.dumps(config_list, indent=4))
        return func_dict[compressor_type](algorithm_name, model, config_list, evaluator, optimizer, trainer,
                                          finetune_trainer, **compressor_parameter_dict)

    @classmethod
    def trial_execute_compress(cls, module_name):
        auto_compress_module = import_('{}.AutoCompressModule'.format(module_name))

        parameter = nni.get_next_parameter()['compressor_type']
        compressor_type, algorithm_config = parameter['_name'], parameter['algorithm_name']
        algorithm_name = algorithm_config['_name']
        converted_config_dict = {k: v for k, v in algorithm_config.items() if k.startswith('config_list::')}
        parameter_dict = {k.split('parameter::')[1]: v for k, v in algorithm_config.items() if k.startswith('parameter::')}

        config_list = cls._convert_config_list(compressor_type, converted_config_dict)

        model, evaluator, optimizer = auto_compress_module.model(), auto_compress_module.evaluator(), auto_compress_module.optimizer()
        trainer = auto_compress_module.trainer(compressor_type, algorithm_name)
        finetune_trainer = auto_compress_module.trainer(compressor_type, algorithm_name)

        compressed_model = cls._compress_pipeline(compressor_type, algorithm_name, model, config_list, evaluator,
                                                  optimizer, trainer, finetune_trainer, **parameter_dict)

        nni.report_final_result(evaluator(compressed_model))
