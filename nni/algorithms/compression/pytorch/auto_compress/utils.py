# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any


class AutoCompressSearchSpaceGenerator:
    """
    For convenient generation of search space that can be used by tuner.
    """

    def __init__(self):
        self.pruner_choice_list = []
        self.quantizer_choice_list = []

    def add_pruner_config(self, pruner_name: str, config_list: list, **algo_kwargs):
        """
        Parameters
        ----------
        pruner_name
            Supported pruner name: 'level', 'slim', 'l1', 'l2', 'fpgm', 'taylorfo', 'apoz', 'mean_activation'.
        config_list
            Except 'op_types' and 'op_names', other config value can be written as `{'_type': ..., '_value': ...}`.
        **algo_kwargs
            The additional pruner parameters except 'model', 'config_list', 'optimizer'.
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
        self.pruner_choice_list.append(sub_search_space)

    def add_quantizer_config(self, quantizer_name: str, config_list: list, **algo_kwargs):
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
        self.quantizer_choice_list.append(sub_search_space)

    def dumps(self) -> dict:
        """
        Dump the search space as a dict.
        """
        compressor_choice_value = []
        if self.pruner_choice_list:
            compressor_choice_value.append({
                '_name': 'pruner',
                'algorithm_name': {
                    '_type': 'choice',
                    '_value': self.pruner_choice_list
                }
            })
        if self.quantizer_choice_list:
            compressor_choice_value.append({
                '_name': 'quantizer',
                'algorithm_name': {
                    '_type': 'choice',
                    '_value': self.quantizer_choice_list
                }
            })
        search_space = {
            'compressor_type': {
                '_type': 'choice',
                '_value': compressor_choice_value
            }
        }
        return search_space

    @classmethod
    def loads(cls, search_space: dict):
        """
        Return a AutoCompressSearchSpaceGenerator instance load from a search space dict.
        """
        generator = AutoCompressSearchSpaceGenerator()
        for v in search_space['compressor_type']['_value']:
            if v['_name'] == 'pruner':
                generator.pruner_choice_list = v['algorithm_name']['_value']
            if v['_name'] == 'quantizer':
                generator.quantizer_choice_list = v['algorithm_name']['_value']
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
