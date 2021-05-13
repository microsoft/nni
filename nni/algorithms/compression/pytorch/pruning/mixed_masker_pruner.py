# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from schema import And, Optional

from nni.compression.pytorch.utils.config_validation import CompressorSchema
from nni.algorithms.compression.pytorch.pruning.one_shot import _StructuredFilterPruner

_logger = logging.getLogger(__name__)


class MixedMaskerPruner(_StructuredFilterPruner):
    """
    MixedMaskerPruner support config different masker in operation level.
    """

    def __init__(self, model, config_list, optimizer=None, dependency_aware=False, dummy_input=None, maskers_config_dict=None):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Model to be pruned
        config_list : list
            Supported keys:
            - sparsity : This is to specify the sparsity operations to be compressed to.
            - op_types : See specific pruner introduction more information.
            - pruning_algo(Optional): A tuple of the type of masker and the args of masker, i.e. ('level', {})
            - masker_name(Optional): If use maskers_config_dict, the value is the key in maskers_config_dict.
        optimizer: torch.optim.Optimizer
            Optimizer used to train model
        dependency_aware: bool
            If use dependency aware mode
        dummy_input: torch.Tensor
            Required in dependency aware mode
        maskers_config_dict: dict
            Reuse pruning_algo value in config_list, key is a custom name of masker and value has the same scheme with pruning_algo in config_list. i.e. {'level_0': ('level', {})}
        """
        if maskers_config_dict is None:
            config_list, maskers_config_dict = self.__convert_config_list(config_list)
        super().__init__(model, config_list, 'mixed', optimizer=optimizer, dependency_aware=dependency_aware,
                         dummy_input=dummy_input, maskers_config_dict=maskers_config_dict)
        _logger.debug('Set MixedMasker successfully.')

    def validate_config(self, model, config_list):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Model to be pruned
        config_list : list
            List on pruning configs
        """
        schema = CompressorSchema([{
            'sparsity': And(float, lambda n: 0 < n < 1),
            Optional('op_types'): [str],
            Optional('op_names'): [str],
            Optional('masker_name'): str
        }], model, _logger)

        schema.validate(config_list)

    def __convert_config_list(self, config_list):
        maskers_config_dict = {}
        counter = {}
        for config in config_list:
            assert 'masker_name' in config, 'maskers_config_dict should be set if use masker_name'
            if 'pruning_algo' not in config:
                config['masker_name'] = 'default'
            else:
                masker_type, _ = config['pruning_algo']
                counter[masker_type] = 1 + counter.get(masker_type, 0)
                masker_name = '{}_{}'.format(masker_type, counter[masker_type])
                maskers_config_dict[masker_name] = config.pop('pruning_algo')
                config['masker_name'] = masker_name
        return config_list, maskers_config_dict
