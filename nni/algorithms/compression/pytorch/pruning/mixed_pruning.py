# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Any
from .weight_masker import WeightMasker
from .structured_pruning import StructuredWeightMasker
from .finegrained_pruning import LevelPrunerMasker
from .structured_pruning import SlimPrunerMasker, L1FilterPrunerMasker, \
    L2FilterPrunerMasker, FPGMPrunerMasker, TaylorFOWeightFilterPrunerMasker, \
    ActivationAPoZRankFilterPrunerMasker, ActivationMeanRankFilterPrunerMasker

__all__ = ['MixedPrunerMasker']

MASKER_DICT = {
    'level': LevelPrunerMasker,
    'slim': SlimPrunerMasker,
    'l1': L1FilterPrunerMasker,
    'l2': L2FilterPrunerMasker,
    'fpgm': FPGMPrunerMasker,
    'taylorfo': TaylorFOWeightFilterPrunerMasker,
    'apoz': ActivationAPoZRankFilterPrunerMasker,
    'mean_activation': ActivationMeanRankFilterPrunerMasker
}

_logger = logging.getLogger('torch pruner')


class MixedPrunerMasker(WeightMasker):
    def __init__(self, model, pruner, maskers_config_dict):
        self.model = model
        self.pruner = pruner
        self.maskers = dict()
        self.dependency_aware = False

        for masker_name, masker_config in maskers_config_dict.items():
            masker_type, masker_args = masker_config
            assert masker_type in MASKER_DICT, 'Unsupported masker type {}.'.format(masker_type)
            self.maskers[masker_name] = MASKER_DICT[masker_type](self.model, self.pruner, **masker_args)

        if 'default' not in self.maskers:
            self.maskers['default'] = MASKER_DICT['level'](self.model, self.pruner)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == 'dependency_aware':
            for masker in self.maskers.values():
                masker.dependency_aware = value if isinstance(masker, StructuredWeightMasker) else masker.dependency_aware
        return super().__setattr__(name, value)

    def calc_mask(self, sparsity, wrapper, wrapper_idx=None, **depen_kwargs):
        masker_name = 'default' if wrapper.config.get('masker_name') is None else wrapper.config['masker_name']
        masker = self.maskers[masker_name]
        if isinstance(masker, StructuredWeightMasker):
            return masker.calc_mask(sparsity, wrapper, wrapper, **depen_kwargs)
        else:
            if depen_kwargs:
                _logger.warning('Submasker type %s not support dependency aware.', type(masker).__name__)
            return masker.calc_mask(sparsity, wrapper, wrapper)
