# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from schema import And, Optional
from nni._graph_utils import TorchModuleGraph
from nni.compression.torch.utils.shape_dependency import ChannelDependency, GroupDependency
from .constants import MASKER_DICT
from ..utils.config_validation import CompressorSchema
from ..compressor import Pruner


__all__ = ['LevelPruner', 'SlimPruner', 'L1FilterPruner', 'L2FilterPruner', 'FPGMPruner', \
    'TaylorFOWeightFilterPruner', 'ActivationAPoZRankFilterPruner', 'ActivationMeanRankFilterPruner', \
    'Constrained_L1FilterPruner', 'Constrained_L2FilterPruner', 'ConstrainedActivationMeanRankFilterPruner']

logger = logging.getLogger('torch pruner')

class OneshotPruner(Pruner):
    """
    Prune model to an exact pruning level for one time.
    """

    def __init__(self, model, config_list, pruning_algorithm='level', optimizer=None, **algo_kwargs):
        """
        Parameters
        ----------
        model : torch.nn.module
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
        self.masker = MASKER_DICT[pruning_algorithm](model, self, **algo_kwargs)

    def validate_config(self, model, config_list):
        """
        Parameters
        ----------
        model : torch.nn.module
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
            masks = self.masker.calc_mask(sparsity=sparsity, wrapper=wrapper, wrapper_idx=wrapper_idx)

            # masker.calc_mask returns None means calc_mask is not calculated sucessfully, can try later
            if masks is not None:
                wrapper.if_calculated = True
            return masks
        else:
            return None

class LevelPruner(OneshotPruner):
    def __init__(self, model, config_list, optimizer=None):
        super().__init__(model, config_list, pruning_algorithm='level', optimizer=optimizer)

class SlimPruner(OneshotPruner):
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
    def __init__(self, model, config_list, pruning_algorithm, optimizer=None, **algo_kwargs):
        super().__init__(model, config_list, pruning_algorithm=pruning_algorithm, optimizer=optimizer, **algo_kwargs)

    def validate_config(self, model, config_list):
        schema = CompressorSchema([{
            'sparsity': And(float, lambda n: 0 < n < 1),
            'op_types': ['Conv2d'],
            Optional('op_names'): [str]
        }], model, logger)

        schema.validate(config_list)

class _Constrained_StructuredFilterPruner(OneshotPruner):
    def __init__(self, model, config_list, dummy_input, pruning_algorithm, optimizer=None, **algo_kwargs):
        super().__init__(model, config_list, pruning_algorithm=pruning_algorithm, optimizer=optimizer, **algo_kwargs)
        # Get the TorchModuleGraph of the target model
        # to trace the model, we need to unwrap the wrappers
        self._unwrap_model()
        self.graph = TorchModuleGraph(model, dummy_input)
        self._wrap_model()
        self.channel_depen = ChannelDependency(traced_model=self.graph.trace)
        self.group_depen = GroupDependency(traced_model=self.graph.trace)
        self.channel_depen = self.channel_depen.dependency_sets
        self.channel_depen = {name : sets for sets in self.channel_depen for name in sets}
        self.group_depen = self.group_depen.dependency_sets

    def validate_config(self, model, config_list):
        schema = CompressorSchema([{
            'sparsity': And(float, lambda n: 0 < n < 1),
            'op_types': ['Conv2d'],
            Optional('op_names'): [str]
        }], model, logger)

        schema.validate(config_list)

    def calc_mask(self, wrappers, channel_dsets, wrappers_idx=None):
        """
        calculate the masks for the conv layers in the same
        channel dependecy set. All the layers passed in have
        the same number of channels.

        Parameters
        ----------
        wrappers : list
            The list of the wrappers that in the same channel dependency
            set.
        wrappers_idx : list
            The list of the indexes of wrapppers.
        """
        # The number of the groups for each conv layers
        # Note that, this number may be different from its
        # original number of groups of filters.
        groups = [self.group_depen[_w.name] for _w in wrappers]
        masks = self.masker.calc_mask(wrappers, channel_dsets, groups, wrappers_idx=wrappers_idx)
        if masks is not None:
            # if masks is None, then the mask calculation fails.
            # for example, in activation related maskers, we should
            # pass enough batches of data to the model, so that the
            # masks can be calculated successfully.
            for _w in wrappers:
                _w.if_calculated = True
        return masks

    def update_mask(self):
        """
        In the original StructuredFilterPruner, the wraper of each layer will update its
        mask own mask according to the sparsity specified in the config_list. However, in
        the _Constrained_StructuredFilterPruner, we may prune several layers at the same
        time according the sparsities and the channel/group depedencies. So we need to
        overwrite this function.
        """
        name2wraper = {x.name: x for x in self.get_modules_wrapper()}
        wraper2index = {x:i for i, x in enumerate(self.get_modules_wrapper())}
        for wrapper in self.get_modules_wrapper():
            if wrapper.if_calculated:
                continue
            # find all the conv layers that have channel dependecy with this layer
            # and prune all these layers at the same time.
            _names = [x for x in self.channel_depen[wrapper.name]]
            logger.info('Pruning the dependent layers: %s', ','.join(_names))
            _wrappers = [name2wraper[name] for name in _names if name in name2wraper]
            _wrapper_idxes = [wraper2index[_w] for _w in _wrappers]

            masks = self.calc_mask(_wrappers, _names, wrappers_idx=_wrapper_idxes)
            if masks is not None:
                for layer in masks:
                    for k in masks[layer]:
                        assert hasattr(name2wraper[layer], k), "there is no attribute '%s' in wrapper on %s" % (k, layer)
                        setattr(name2wraper[layer], k, masks[layer][k])

    def compress(self):
        """
        TODO:
        To avoid retraining the BatchNorm of model after the speedup progress,
        we should also infer the mask at the end of the compress function.
        """
        self.update_mask()
        return self.bound_model

class L1FilterPruner(_StructuredFilterPruner):
    def __init__(self, model, config_list, optimizer=None):
        super().__init__(model, config_list, pruning_algorithm='l1', optimizer=optimizer)

class Constrained_L1FilterPruner(_Constrained_StructuredFilterPruner):
    def __init__(self, model, config_list, dummy_input, optimizer=None):
        """
        A one-shot pruner that prunes the model according to the l2-norm
        and the channel-dependency/group-dependency of the model. Because
        this is a topology constraint aware pruner, so the speedup module
        can better harvest the speed benefit from the pruned model.

        Paramters
        ---------
        model : torch.nn.Module
            The target model to be pruned.
        config_list : list
            The configs that specify the sparsity for each layer.
        dummy_input : torch.Tensor
            The dummy input to analyze the topology constraints.
        optimizer : torch.optim.Optimizer
            The Optimizer used to train the model(if needed).
        """
        super().__init__(model, config_list, dummy_input, pruning_algorithm='l1_constrained', optimizer=optimizer)

class L2FilterPruner(_StructuredFilterPruner):
    def __init__(self, model, config_list, optimizer=None):
        super().__init__(model, config_list, pruning_algorithm='l2', optimizer=optimizer)

class Constrained_L2FilterPruner(_Constrained_StructuredFilterPruner):
    def __init__(self, model, config_list, dummy_input, optimizer=None):
        """
        A one-shot pruner that prunes the model according to the l2-norm
        and the channel-dependency/group-dependency of the model. Because
        this is a topology constraint aware pruner, so the speedup module
        can better harvest the speed benefit from the pruned model.

        Paramters
        ---------
        model : torch.nn.Module
            The target model to be pruned.
        config_list : list
            The configs that specify the sparsity for each layer.
        dummy_input : torch.Tensor
            The dummy input to analyze the topology constraints.
        optimizer : torch.optim.Optimizer
            The Optimizer used to train the model(if needed).

        """
        super().__init__(model, config_list, dummy_input, pruning_algorithm='l2_constrained', optimizer=optimizer)

class FPGMPruner(_StructuredFilterPruner):
    def __init__(self, model, config_list, optimizer=None):
        super().__init__(model, config_list, pruning_algorithm='fpgm', optimizer=optimizer)

class TaylorFOWeightFilterPruner(_StructuredFilterPruner):
    def __init__(self, model, config_list, optimizer=None, statistics_batch_num=1):
        super().__init__(model, config_list, pruning_algorithm='taylorfo', optimizer=optimizer, statistics_batch_num=statistics_batch_num)

class ActivationAPoZRankFilterPruner(_StructuredFilterPruner):
    def __init__(self, model, config_list, optimizer=None, activation='relu', statistics_batch_num=1):
        super().__init__(model, config_list, pruning_algorithm='apoz', optimizer=optimizer, \
            activation=activation, statistics_batch_num=statistics_batch_num)

class ActivationMeanRankFilterPruner(_StructuredFilterPruner):
    def __init__(self, model, config_list, optimizer=None, activation='relu', statistics_batch_num=1):
        super().__init__(model, config_list, pruning_algorithm='mean_activation', optimizer=optimizer, \
            activation=activation, statistics_batch_num=statistics_batch_num)

class ConstrainedActivationMeanRankFilterPruner(_Constrained_StructuredFilterPruner):
    def __init__(self, model, config_list, dummy_input, optimizer=None, activation='relu', statistics_batch_num=1):
        super().__init__(model, config_list, dummy_input, pruning_algorithm='mean_activation_constrained', optimizer=optimizer, \
            activation=activation, statistics_batch_num=statistics_batch_num)
