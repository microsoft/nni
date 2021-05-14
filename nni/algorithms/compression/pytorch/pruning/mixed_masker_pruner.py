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
            assert 'masker_name' not in config, 'maskers_config_dict should be set if use masker_name'
            if 'pruning_algo' not in config:
                config['masker_name'] = 'default'
            else:
                masker_type, _ = config['pruning_algo']
                counter[masker_type] = 1 + counter.get(masker_type, 0)
                masker_name = '{}_{}'.format(masker_type, counter[masker_type])
                maskers_config_dict[masker_name] = config.pop('pruning_algo')
                config['masker_name'] = masker_name
        return config_list, maskers_config_dict

    def _dependency_calc_mask(self, wrappers, channel_dsets, wrappers_idx=None, origin_wrapper=None):
        """
        Override to make sure if submasker do not support dependency_aware, calc_mask perform normal calculations.
        """
        groups = [self.group_depen[_w.name] for _w in wrappers]
        sparsities = [_w.config['sparsity'] for _w in wrappers]
        masks = self.masker.calc_mask(
            sparsities, wrappers, wrappers_idx, channel_dsets=channel_dsets, groups=groups, origin_wrapper=origin_wrapper)
        if masks is not None:
            if isinstance(masks, dict):
                for _w in wrappers:
                    _w.if_calculated = True
            else:
                origin_wrapper.if_calculated = True
                masks = {origin_wrapper.name: masks}
        return masks

    def _common_calc_mask(self, wrapper, wrappers_idx=None):
        return self.masker.calc_mask(wrapper.config['sparsity'], wrapper, wrappers_idx=wrappers_idx)

    def _dependency_update_mask(self):
        """
        Override to make sure if submasker do not support dependency_aware, calc_mask perform normal calculations.
        """
        name2wrapper = {x.name: x for x in self.get_modules_wrapper()}
        wrapper2index = {x: i for i, x in enumerate(self.get_modules_wrapper())}
        for wrapper in self.get_modules_wrapper():
            if wrapper.if_calculated:
                continue
            if wrapper.name not in self.channel_depen:
                masks = self._common_calc_mask(wrapper, wrapper2index[wrapper])
                if masks is not None:
                    for k in masks:
                        assert hasattr(wrapper, k), "there is no attribute '%s' in wrapper" % k
                        setattr(wrapper, k, masks[k])
            else:
                _names = [x for x in self.channel_depen[wrapper.name]]
                _wrappers = [name2wrapper[name] for name in _names if name in name2wrapper]
                _wrapper_idxes = [wrapper2index[_w] for _w in _wrappers]

                masks = self._dependency_calc_mask(_wrappers, _names, wrappers_idx=_wrapper_idxes, origin_wrapper=wrapper)
                if masks is not None:
                    for layer in masks:
                        for mask_type in masks[layer]:
                            assert hasattr(
                                name2wrapper[layer], mask_type), "there is no attribute '%s' in wrapper on %s" % (mask_type, layer)
                            setattr(name2wrapper[layer], mask_type, masks[layer][mask_type])
