# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

class AutoCompressSearchSpaceGenerator:
    def __init__(self):
        self.pruner_choice_list = []
        self.quantizer_choice_list = []

    def add_pruner_config(self, pruner_name: str, config_list: dict, **algo_kwargs):
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

    def add_quantizer_config(self, quantizer_name: str, config_list: dict, **algo_kwargs):
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

    def dumps(self) -> str:
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

    def loads(self, search_space: dict):
        for v in search_space['compressor_type']['_value']:
            if v['_name'] == 'pruner':
                self.pruner_choice_list = v['algorithm_name']['_value']
            if v['_name'] == 'quantizer':
                self.quantizer_choice_list = v['algorithm_name']['_value']

    def _wrap_single_value(self, value) -> dict:
        if not isinstance(value, dict) or '_type' not in value:
            value = {
                '_type': 'choice',
                '_value': [value]
            }
        return value
