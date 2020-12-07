# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from schema import And, Optional, SchemaError
from nni.common.graph_utils import TorchModuleGraph
from nni.compression.pytorch.utils.shape_dependency import ChannelDependency, GroupDependency
from .constants import MASKER_DICT
from nni.compression.pytorch.utils.config_validation import CompressorSchema
from nni.compression.pytorch.compressor import Pruner


__all__ = ['LevelPruner', 'SlimPruner', 'L1FilterPruner', 'L2FilterPruner', 'FPGMPruner',
           'TaylorFOWeightFilterPruner', 'ActivationAPoZRankFilterPruner', 'ActivationMeanRankFilterPruner']

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OneshotPruner(Pruner):
    """
    Prune model to an exact pruning level for one time.
    """

    def __init__(self, model, config_list, pruning_algorithm='level', optimizer=None, **algo_kwargs):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Model to be pruned
        config_list : list
            List on pruning configs
        pruning_algorithm: str
            algorithms being used to prune model
        optimizer: torch.optim.Optimizer
            Optimizer used to train model
        algo_kwargs: dict
            Additional parameters passed to pruning algorithm masker class
        """

        super().__init__(model, config_list, optimizer)
        self.set_wrappers_attribute("if_calculated", False)
        self.masker = MASKER_DICT[pruning_algorithm](
            model, self, **algo_kwargs)

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
            Optional('op_names'): [str]
        }], model, logger)

        schema.validate(config_list)

    def calc_mask(self, wrapper, wrapper_idx=None):
        """
        Calculate the mask of given layer
        Parameters
        ----------
        wrapper : Module
            the module to instrument the compression operation
        wrapper_idx: int
            index of this wrapper in pruner's all wrappers
        Returns
        -------
        dict
            dictionary for storing masks, keys of the dict:
            'weight_mask':  weight mask tensor
            'bias_mask': bias mask tensor (optional)
        """
        if wrapper.if_calculated:
            return None

        sparsity = wrapper.config['sparsity']
        if not wrapper.if_calculated:
            masks = self.masker.calc_mask(
                sparsity=sparsity, wrapper=wrapper, wrapper_idx=wrapper_idx)

            # masker.calc_mask returns None means calc_mask is not calculated sucessfully, can try later
            if masks is not None:
                wrapper.if_calculated = True
            return masks
        else:
            return None


class LevelPruner(OneshotPruner):
    """
    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned
    config_list : list
        Supported keys:
            - sparsity : This is to specify the sparsity operations to be compressed to.
            - op_types : Operation types to prune.
    optimizer: torch.optim.Optimizer
            Optimizer used to train model
    """

    def __init__(self, model, config_list, optimizer=None):
        super().__init__(model, config_list, pruning_algorithm='level', optimizer=optimizer)


class SlimPruner(OneshotPruner):
    """
    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned
    config_list : list
        Supported keys:
            - sparsity : This is to specify the sparsity operations to be compressed to.
            - op_types : Only BatchNorm2d is supported in Slim Pruner.
    optimizer: torch.optim.Optimizer
            Optimizer used to train model
    """

    def __init__(self, model, config_list, optimizer=None):
        super().__init__(model, config_list, pruning_algorithm='slim', optimizer=optimizer)

    def validate_config(self, model, config_list):
        schema = CompressorSchema([{
            'sparsity': And(float, lambda n: 0 < n < 1),
            'op_types': ['BatchNorm2d'],
            Optional('op_names'): [str]
        }], model, logger)

        schema.validate(config_list)

        if len(config_list) > 1:
            logger.warning('Slim pruner only supports 1 configuration')


class _StructuredFilterPruner(OneshotPruner):
    """
    _StructuredFilterPruner has two ways to calculate the masks
    for conv layers. In the normal way, the _StructuredFilterPruner
    will calculate the mask of each layer separately. For example, each
    conv layer determine which filters should be pruned according to its L1
    norm. In constrast, in the dependency-aware way, the layers that in a
    dependency group will be pruned jointly and these layers will be forced
    to prune the same channels.
    """

    def __init__(self, model, config_list, pruning_algorithm, optimizer=None, dependency_aware=False, dummy_input=None, **algo_kwargs):
        super().__init__(model, config_list, pruning_algorithm=pruning_algorithm,
                         optimizer=optimizer, **algo_kwargs)
        self.dependency_aware = dependency_aware
        # set the dependency-aware switch for the masker
        self.masker.dependency_aware = dependency_aware
        self.dummy_input = dummy_input
        if self.dependency_aware:
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

    def update_mask(self):
        if not self.dependency_aware:
            # if we use the normal way to update the mask,
            # then call the update_mask of the father class
            super(_StructuredFilterPruner, self).update_mask()
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
                        assert hasattr(
                            name2wrapper[layer], mask_type), "there is no attribute '%s' in wrapper on %s" % (mask_type, layer)
                        setattr(name2wrapper[layer], mask_type, masks[layer][mask_type])


class L1FilterPruner(_StructuredFilterPruner):
    """
    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned
    config_list : list
        Supported keys:
            - sparsity : This is to specify the sparsity operations to be compressed to.
            - op_types : Only Conv2d is supported in L1FilterPruner.
    optimizer: torch.optim.Optimizer
            Optimizer used to train model
    dependency_aware: bool
        If prune the model in a dependency-aware way. If it is `True`, this pruner will
        prune the model according to the l2-norm of weights and the channel-dependency or
        group-dependency of the model. In this way, the pruner will force the conv layers
        that have dependencies to prune the same channels, so the speedup module can better
        harvest the speed benefit from the pruned model. Note that, if this flag is set True
        , the dummy_input cannot be None, because the pruner needs a dummy input to trace the
        dependency between the conv layers.
    dummy_input : torch.Tensor
        The dummy input to analyze the topology constraints. Note that, the dummy_input
        should on the same device with the model.
    """

    def __init__(self, model, config_list, optimizer=None, dependency_aware=False, dummy_input=None):
        super().__init__(model, config_list, pruning_algorithm='l1', optimizer=optimizer,
                         dependency_aware=dependency_aware, dummy_input=dummy_input)


class L2FilterPruner(_StructuredFilterPruner):
    """
    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned
    config_list : list
        Supported keys:
            - sparsity : This is to specify the sparsity operations to be compressed to.
            - op_types : Only Conv2d is supported in L2FilterPruner.
    optimizer: torch.optim.Optimizer
            Optimizer used to train model
    dependency_aware: bool
        If prune the model in a dependency-aware way. If it is `True`, this pruner will
        prune the model according to the l2-norm of weights and the channel-dependency or
        group-dependency of the model. In this way, the pruner will force the conv layers
        that have dependencies to prune the same channels, so the speedup module can better
        harvest the speed benefit from the pruned model. Note that, if this flag is set True
        , the dummy_input cannot be None, because the pruner needs a dummy input to trace the
        dependency between the conv layers.
    dummy_input : torch.Tensor
        The dummy input to analyze the topology constraints. Note that, the dummy_input
        should on the same device with the model.
    """

    def __init__(self, model, config_list, optimizer=None, dependency_aware=False, dummy_input=None):
        super().__init__(model, config_list, pruning_algorithm='l2', optimizer=optimizer,
                         dependency_aware=dependency_aware, dummy_input=dummy_input)


class FPGMPruner(_StructuredFilterPruner):
    """
    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned
    config_list : list
        Supported keys:
            - sparsity : This is to specify the sparsity operations to be compressed to.
            - op_types : Only Conv2d is supported in FPGM Pruner.
    optimizer: torch.optim.Optimizer
            Optimizer used to train model
    dependency_aware: bool
        If prune the model in a dependency-aware way. If it is `True`, this pruner will
        prune the model according to the l2-norm of weights and the channel-dependency or
        group-dependency of the model. In this way, the pruner will force the conv layers
        that have dependencies to prune the same channels, so the speedup module can better
        harvest the speed benefit from the pruned model. Note that, if this flag is set True
        , the dummy_input cannot be None, because the pruner needs a dummy input to trace the
        dependency between the conv layers.
    dummy_input : torch.Tensor
        The dummy input to analyze the topology constraints. Note that, the dummy_input
        should on the same device with the model.
    """

    def __init__(self, model, config_list, optimizer=None, dependency_aware=False, dummy_input=None):
        super().__init__(model, config_list, pruning_algorithm='fpgm',
                         dependency_aware=dependency_aware, dummy_input=dummy_input, optimizer=optimizer)


class TaylorFOWeightFilterPruner(_StructuredFilterPruner):
    """
    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned
    config_list : list
        Supported keys:
            - sparsity : How much percentage of convolutional filters are to be pruned.
            - op_types : Currently only Conv2d is supported in TaylorFOWeightFilterPruner.
    optimizer: torch.optim.Optimizer
            Optimizer used to train model
    statistics_batch_num: int
        The number of batches to statistic the activation.
    dependency_aware: bool
        If prune the model in a dependency-aware way. If it is `True`, this pruner will
        prune the model according to the l2-norm of weights and the channel-dependency or
        group-dependency of the model. In this way, the pruner will force the conv layers
        that have dependencies to prune the same channels, so the speedup module can better
        harvest the speed benefit from the pruned model. Note that, if this flag is set True
        , the dummy_input cannot be None, because the pruner needs a dummy input to trace the
        dependency between the conv layers.
    dummy_input : torch.Tensor
        The dummy input to analyze the topology constraints. Note that, the dummy_input
        should on the same device with the model.

    """

    def __init__(self, model, config_list, optimizer=None, statistics_batch_num=1,
                 dependency_aware=False, dummy_input=None):
        super().__init__(model, config_list, pruning_algorithm='taylorfo',
                         dependency_aware=dependency_aware, dummy_input=dummy_input,
                         optimizer=optimizer, statistics_batch_num=statistics_batch_num)


class ActivationAPoZRankFilterPruner(_StructuredFilterPruner):
    """
    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned
    config_list : list
        Supported keys:
            - sparsity : How much percentage of convolutional filters are to be pruned.
            - op_types : Only Conv2d is supported in ActivationAPoZRankFilterPruner.
    optimizer: torch.optim.Optimizer
            Optimizer used to train model
    activation: str
        The activation type.
    statistics_batch_num: int
        The number of batches to statistic the activation.
    dependency_aware: bool
        If prune the model in a dependency-aware way. If it is `True`, this pruner will
        prune the model according to the l2-norm of weights and the channel-dependency or
        group-dependency of the model. In this way, the pruner will force the conv layers
        that have dependencies to prune the same channels, so the speedup module can better
        harvest the speed benefit from the pruned model. Note that, if this flag is set True
        , the dummy_input cannot be None, because the pruner needs a dummy input to trace the
        dependency between the conv layers.
    dummy_input : torch.Tensor
        The dummy input to analyze the topology constraints. Note that, the dummy_input
        should on the same device with the model.

    """

    def __init__(self, model, config_list, optimizer=None, activation='relu',
                 statistics_batch_num=1, dependency_aware=False, dummy_input=None):
        super().__init__(model, config_list, pruning_algorithm='apoz', optimizer=optimizer,
                         dependency_aware=dependency_aware, dummy_input=dummy_input,
                         activation=activation, statistics_batch_num=statistics_batch_num)


class ActivationMeanRankFilterPruner(_StructuredFilterPruner):
    """
    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned
    config_list : list
        Supported keys:
            - sparsity : How much percentage of convolutional filters are to be pruned.
            - op_types : Only Conv2d is supported in ActivationMeanRankFilterPruner.
    optimizer: torch.optim.Optimizer
            Optimizer used to train model.
    activation: str
        The activation type.
    statistics_batch_num: int
        The number of batches to statistic the activation.
    dependency_aware: bool
        If prune the model in a dependency-aware way. If it is `True`, this pruner will
        prune the model according to the l2-norm of weights and the channel-dependency or
        group-dependency of the model. In this way, the pruner will force the conv layers
        that have dependencies to prune the same channels, so the speedup module can better
        harvest the speed benefit from the pruned model. Note that, if this flag is set True
        , the dummy_input cannot be None, because the pruner needs a dummy input to trace the
        dependency between the conv layers.
    dummy_input : torch.Tensor
        The dummy input to analyze the topology constraints. Note that, the dummy_input
        should on the same device with the model.
    """

    def __init__(self, model, config_list, optimizer=None, activation='relu',
                 statistics_batch_num=1, dependency_aware=False, dummy_input=None):
        super().__init__(model, config_list, pruning_algorithm='mean_activation', optimizer=optimizer,
                         dependency_aware=dependency_aware, dummy_input=dummy_input,
                         activation=activation, statistics_batch_num=statistics_batch_num)
