# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from schema import And, Optional, SchemaError
from nni.common.graph_utils import TorchModuleGraph
from nni.compression.pytorch.utils.shape_dependency import ChannelDependency, GroupDependency
from nni.compression.pytorch.utils.config_validation import CompressorSchema
from nni.compression.pytorch.compressor import Pruner
from .constants import MASKER_DICT

__all__ = ['DependencyAwarePruner']

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DependencyAwarePruner(Pruner):
    """
    DependencyAwarePruner has two ways to calculate the masks
    for conv layers. In the normal way, the DependencyAwarePruner
    will calculate the mask of each layer separately. For example, each
    conv layer determine which filters should be pruned according to its L1
    norm. In constrast, in the dependency-aware way, the layers that in a
    dependency group will be pruned jointly and these layers will be forced
    to prune the same channels.
    """

    def __init__(self, model, config_list, optimizer=None, pruning_algorithm='level', dependency_aware=False,
                 dummy_input=None, **algo_kwargs):
        super().__init__(model, config_list=config_list, optimizer=optimizer)

        self.dependency_aware = dependency_aware
        self.dummy_input = dummy_input

        if self.dependency_aware:
            if not self._supported_dependency_aware():
                raise ValueError('This pruner does not support dependency aware!')

            errmsg = "When dependency_aware is set, the dummy_input should not be None"
            assert self.dummy_input is not None, errmsg
            # Get the TorchModuleGraph of the target model
            # to trace the model, we need to unwrap the wrappers
            self._unwrap_model()
            self.graph = TorchModuleGraph(model, dummy_input)
            self._wrap_model()
            self.channel_depen = ChannelDependency(
                traced_model=self.graph.trace)
            self.group_depen = GroupDependency(traced_model=self.graph.trace)
            self.channel_depen = self.channel_depen.dependency_sets
            self.channel_depen = {
                name: sets for sets in self.channel_depen for name in sets}
            self.group_depen = self.group_depen.dependency_sets

        self.masker = MASKER_DICT[pruning_algorithm](
            model, self, **algo_kwargs)
        # set the dependency-aware switch for the masker
        self.masker.dependency_aware = dependency_aware
        self.set_wrappers_attribute("if_calculated", False)

    def calc_mask(self, wrapper, wrapper_idx=None):
        if not wrapper.if_calculated:
            sparsity = wrapper.config['sparsity']
            masks = self.masker.calc_mask(
                sparsity=sparsity, wrapper=wrapper, wrapper_idx=wrapper_idx)

            # masker.calc_mask returns None means calc_mask is not calculated sucessfully, can try later
            if masks is not None:
                wrapper.if_calculated = True
            return masks
        else:
            return None

    def update_mask(self):
        if not self.dependency_aware:
            # if we use the normal way to update the mask,
            # then call the update_mask of the father class
            super(DependencyAwarePruner, self).update_mask()
        else:
            # if we update the mask in a dependency-aware way
            # then we call _dependency_update_mask
            self._dependency_update_mask()

    def validate_config(self, model, config_list):
        schema = CompressorSchema([{
            Optional('sparsity'): And(float, lambda n: 0 < n < 1),
            Optional('op_types'): ['Conv2d'],
            Optional('op_names'): [str],
            Optional('exclude'): bool
        }], model, logger)

        schema.validate(config_list)
        for config in config_list:
            if 'exclude' not in config and 'sparsity' not in config:
                raise SchemaError('Either sparisty or exclude must be specified!')

    def _supported_dependency_aware(self):
        raise NotImplementedError

    def _dependency_calc_mask(self, wrappers, channel_dsets, wrappers_idx=None):
        """
        calculate the masks for the conv layers in the same
        channel dependecy set. All the layers passed in have
        the same number of channels.

        Parameters
        ----------
        wrappers: list
            The list of the wrappers that in the same channel dependency
            set.
        wrappers_idx: list
            The list of the indexes of wrapppers.
        Returns
        -------
        masks: dict
            A dict object that contains the masks of the layers in this
            dependency group, the key is the name of the convolutional layers.
        """
        # The number of the groups for each conv layers
        # Note that, this number may be different from its
        # original number of groups of filters.
        groups = [self.group_depen[_w.name] for _w in wrappers]
        sparsities = [_w.config['sparsity'] for _w in wrappers]
        masks = self.masker.calc_mask(
            sparsities, wrappers, wrappers_idx, channel_dsets=channel_dsets, groups=groups)
        if masks is not None:
            # if masks is None, then the mask calculation fails.
            # for example, in activation related maskers, we should
            # pass enough batches of data to the model, so that the
            # masks can be calculated successfully.
            for _w in wrappers:
                _w.if_calculated = True
        return masks

    def _dependency_update_mask(self):
        """
        In the original update_mask, the wraper of each layer will update its
        own mask according to the sparsity specified in the config_list. However, in
        the _dependency_update_mask, we may prune several layers at the same
        time according the sparsities and the channel/group dependencies.
        """
        name2wrapper = {x.name: x for x in self.get_modules_wrapper()}
        wrapper2index = {x: i for i, x in enumerate(self.get_modules_wrapper())}
        for wrapper in self.get_modules_wrapper():
            if wrapper.if_calculated:
                continue
            # find all the conv layers that have channel dependecy with this layer
            # and prune all these layers at the same time.
            _names = [x for x in self.channel_depen[wrapper.name]]
            logger.info('Pruning the dependent layers: %s', ','.join(_names))
            _wrappers = [name2wrapper[name]
                         for name in _names if name in name2wrapper]
            _wrapper_idxes = [wrapper2index[_w] for _w in _wrappers]

            masks = self._dependency_calc_mask(
                _wrappers, _names, wrappers_idx=_wrapper_idxes)
            if masks is not None:
                for layer in masks:
                    for mask_type in masks[layer]:
                        assert hasattr(name2wrapper[layer], mask_type), "there is no attribute '%s' in wrapper on %s" \
                            % (mask_type, layer)
                        setattr(name2wrapper[layer], mask_type, masks[layer][mask_type])
