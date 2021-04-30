# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import torch

from schema import And, Optional
from nni.compression.pytorch.utils.config_validation import CompressorSchema

from nni.algorithms.compression.pytorch.pruning.constants import MASKER_DICT
from nni.algorithms.compression.pytorch.pruning.one_shot import _StructuredFilterPruner
from nni.algorithms.compression.pytorch.pruning.weight_masker import WeightMasker
from nni.algorithms.compression.pytorch.pruning.structured_pruning import StructuredWeightMasker


_logger = logging.getLogger(__name__)

class MixedMasker(WeightMasker):
    def __init__(self, model, pruner, maskers_config_dict):
        self.model = model
        self.pruner = pruner
        self.maskers = dict()

        for masker_name, masker_config in maskers_config_dict.items():
            masker_type, masker_args = masker_config
            assert masker_type in MASKER_DICT, 'Unsupported masker type {}.'.format(masker_type)
            self.maskers[masker_name] = MASKER_DICT[masker_type](self.model, self.pruner, **masker_args)

        if 'default' not in self.maskers:
            self.maskers['default'] = MASKER_DICT['level'](self.model, self.pruner)

    def calc_mask(self, sparsity, wrapper, wrapper_idx=None, **depen_kwargs):
        masker_name = 'default' if wrapper.config.get('masker_name') is None else wrapper.config['masker_name']
        masker = self.maskers[masker_name]
        if isinstance(masker, StructuredWeightMasker):
            return masker.calc_mask(sparsity, wrapper, wrapper, **depen_kwargs)
        else:
            if not depen_kwargs:
                _logger.warning('Submasker type {} not support dependency aware.'.format(type(masker).__name__))
            return masker.calc_mask(sparsity, wrapper, wrapper)


class MixedMaskerPruner(_StructuredFilterPruner):
    def __init__(self, model, config_list, maskers_config_dict, optimizer=None, dependency_aware=False, dummy_input=None, **algo_kwargs):
        super().__init__(model, config_list, 'level', optimizer=optimizer, dependency_aware=dependency_aware,
                         dummy_input=dummy_input)
        self.masker = MixedMasker(model, self, maskers_config_dict)

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
            Optional('masker_name'): [str]
        }], model, _logger)

        schema.validate(config_list)
