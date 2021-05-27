# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any

from .constants import PRUNER_DICT, QUANTIZER_DICT


class AutoCompressionSearchSpaceGenerator:
    """
    For convenient generation of search space that can be used by tuner.
    """

    def __init__(self):
        self.algorithm_choice_list = []

    def add_config(self, algorithm_name: str, config_list: list, **algo_kwargs):
        """
        This function used for distinguish algorithm type is pruning or quantization.
        Then call `self._add_pruner_config()` or `self._add_quantizer_config()`.
        """
        if algorithm_name in PRUNER_DICT:
            self._add_pruner_config(algorithm_name, config_list, **algo_kwargs)
        if algorithm_name in QUANTIZER_DICT:
            self._add_quantizer_config(algorithm_name, config_list, **algo_kwargs)

    def _add_pruner_config(self, pruner_name: str, config_list: list, **algo_kwargs):
        """
        Parameters
        ----------
        pruner_name
            Supported pruner name: 'level', 'slim', 'l1', 'l2', 'fpgm', 'taylorfo', 'apoz', 'mean_activation'.
        config_list
            Except 'op_types' and 'op_names', other config value can be written as `{'_type': ..., '_value': ...}`.
        **algo_kwargs
            The additional pruner parameters except 'model', 'config_list', 'optimizer', 'trainer', 'criterion'.
            i.e., you can set `statistics_batch_num={'_type': 'choice', '_value': [1, 2, 3]}` in TaylorFOWeightFilterPruner or just `statistics_batch_num=1`.
        """
        sub_search_space = {'_name': pruner_name}
        for config in config_list:
            op_types = config.pop('op_types', [])
            op_names = config.pop('op_names', [])
            key_prefix = 'config_list::{}::{}'.format(':'.join(op_types), ':'.join(op_names))
            for var_name, var_search_space in config.items():
                sub_search_space['{}::{}'.format(key_prefix, var_name)] = self._wrap_single_value(var_search_space)
        for parameter_name, parameter_search_space in algo_kwargs.items():
            key_prefix = 'parameter'
            sub_search_space['{}::{}'.format(key_prefix, parameter_name)] = self._wrap_single_value(parameter_search_space)
        self.algorithm_choice_list.append(sub_search_space)

    def _add_quantizer_config(self, quantizer_name: str, config_list: list, **algo_kwargs):
        """
        Parameters
        ----------
        quantizer_name
            Supported pruner name: 'naive', 'qat', 'dorefa', 'bnn'.
        config_list
            Except 'quant_types', 'op_types' and 'op_names', other config value can be written as `{'_type': ..., '_value': ...}`.
        **algo_kwargs
            The additional pruner parameters except 'model', 'config_list', 'optimizer'.
        """
        sub_search_space = {'_name': quantizer_name}
        for config in config_list:
            quant_types = config.pop('quant_types', [])
            op_types = config.pop('op_types', [])
            op_names = config.pop('op_names', [])
            key_prefix = 'config_list::{}::{}::{}'.format(':'.join(quant_types), ':'.join(op_types), ':'.join(op_names))
            for var_name, var_search_space in config.items():
                sub_search_space['{}::{}'.format(key_prefix, var_name)] = self._wrap_single_value(var_search_space)
        for parameter_name, parameter_search_space in algo_kwargs.items():
            key_prefix = 'parameter'
            sub_search_space['{}::{}'.format(key_prefix, parameter_name)] = self._wrap_single_value(parameter_search_space)
        self.algorithm_choice_list.append(sub_search_space)

    def dumps(self) -> dict:
        """
        Dump the search space as a dict.
        """
        search_space = {
            'algorithm_name': {
                '_type': 'choice',
                '_value': self.algorithm_choice_list
            }
        }
        return search_space

    @classmethod
    def loads(cls, search_space: dict):
        """
        Return a AutoCompressionSearchSpaceGenerator instance load from a search space dict.
        """
        generator = AutoCompressionSearchSpaceGenerator()
        generator.algorithm_choice_list = search_space['algorithm_name']['_value']
        return generator

    def _wrap_single_value(self, value) -> dict:
        if not isinstance(value, dict):
            converted_value = {
                '_type': 'choice',
                '_value': [value]
            }
        elif '_type' not in value:
            converted_value = {}
            for k, v in value.items():
                converted_value[k] = self._wrap_single_value(v)
        else:
            converted_value = value
        return converted_value


def import_(target: str, allow_none: bool = False) -> Any:
    if target is None:
        return None
    path, identifier = target.rsplit('.', 1)
    module = __import__(path, globals(), locals(), [identifier])
    return getattr(module, identifier)
