# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import torch
from .compressor import Pruner

__all__ = ['ActivationAPoZRankFilterPruner', 'ActivationMeanRankFilterPruner']

logger = logging.getLogger('torch activation rank filter pruners')

class ActivationRankFilterPruner(Pruner):
    """
    A structured pruning base class that prunes the filters with the smallest
    importance criterion in convolution layers (using activation values)
    to achieve a preset level of network sparsity.
    """

    def __init__(self, model, config_list, optimizer=None, activation='relu', statistics_batch_num=1):
        """
        Parameters
        ----------
        model : torch.nn.module
            Model to be pruned
        config_list : list
            support key for each list item:
                - sparsity: percentage of convolutional filters to be pruned.
        optimizer: torch.optim.Optimizer
            Optimizer used to train model
        activation : str
            Activation function
        statistics_batch_num : int
            Num of batches for activation statistics
        """

        super().__init__(model, config_list, optimizer)
        self.set_wrappers_attribute("if_calculated", False)
        self.set_wrappers_attribute("collected_activation", [])
        self.statistics_batch_num = statistics_batch_num

        def collector(module_, input_, output):
            if len(module_.collected_activation) < self.statistics_batch_num:
                module_.collected_activation.append(self.activation(output.detach().cpu()))
        self.add_activation_collector(collector)
        assert activation in ['relu', 'relu6']
        if activation == 'relu':
            self.activation = torch.nn.functional.relu
        elif activation == 'relu6':
            self.activation = torch.nn.functional.relu6
        else:
            self.activation = None

    def get_mask(self, base_mask, activations, num_prune):
        raise NotImplementedError('{} get_mask is not implemented'.format(self.__class__.__name__))

    def calc_mask(self, wrapper, **kwargs):
        """
        Calculate the mask of given layer.
        Filters with the smallest importance criterion which is calculated from the activation are masked.

        Parameters
        ----------
        layer : LayerInfo
            the layer to instrument the compression operation
        config : dict
            layer's pruning config

        Returns
        -------
        dict
            dictionary for storing masks
        """

        weight = wrapper.module.weight.data
        op_type = wrapper.type
        config = wrapper.config
        assert 0 <= config.get('sparsity') < 1, "sparsity must in the range [0, 1)"
        assert op_type in ['Conv2d'], "only support Conv2d"
        assert op_type in config.get('op_types')

        if wrapper.if_calculated:
            return None
        mask_weight = torch.ones(weight.size()).type_as(weight).detach()
        if hasattr(wrapper.module, 'bias') and wrapper.module.bias is not None:
            mask_bias = torch.ones(wrapper.module.bias.size()).type_as(wrapper.module.bias).detach()
        else:
            mask_bias = None
        mask = {'weight_mask': mask_weight, 'bias_mask': mask_bias}
        try:
            filters = weight.size(0)
            num_prune = int(filters * config.get('sparsity'))
            if filters < 2 or num_prune < 1 or len(wrapper.collected_activation) < self.statistics_batch_num:
                return mask
            mask = self.get_mask(mask, wrapper.collected_activation, num_prune)
        finally:
            if len(wrapper.collected_activation) == self.statistics_batch_num:
                wrapper.if_calculated = True
        return mask


class ActivationAPoZRankFilterPruner(ActivationRankFilterPruner):
    """
    A structured pruning algorithm that prunes the filters with the
    smallest APoZ(average percentage of zeros) of output activations.
    Hengyuan Hu, Rui Peng, Yu-Wing Tai and Chi-Keung Tang,
    "Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient Deep Architectures", ICLR 2016.
    https://arxiv.org/abs/1607.03250
    """

    def __init__(self, model, config_list, optimizer=None, activation='relu', statistics_batch_num=1):
        """
        Parameters
        ----------
        model : torch.nn.module
            Model to be pruned
        config_list : list
            support key for each list item:
                - sparsity: percentage of convolutional filters to be pruned.
        optimizer: torch.optim.Optimizer
            Optimizer used to train model
        activation : str
            Activation function
        statistics_batch_num : int
            Num of batches for activation statistics
        """
        super().__init__(model, config_list, optimizer, activation, statistics_batch_num)

    def get_mask(self, base_mask, activations, num_prune):
        """
        Calculate the mask of given layer.
        Filters with the smallest APoZ(average percentage of zeros) of output activations are masked.

        Parameters
        ----------
        base_mask : dict
            The basic mask with the same shape of weight, all item in the basic mask is 1.
        activations : list
            Layer's output activations
        num_prune : int
            Num of filters to prune

        Returns
        -------
        dict
            dictionary for storing masks
        """
        apoz = self._calc_apoz(activations)
        prune_indices = torch.argsort(apoz, descending=True)[:num_prune]
        for idx in prune_indices:
            base_mask['weight_mask'][idx] = 0.
            if base_mask['bias_mask'] is not None:
                base_mask['bias_mask'][idx] = 0.
        return base_mask

    def _calc_apoz(self, activations):
        """
        Calculate APoZ(average percentage of zeros) of activations.

        Parameters
        ----------
        activations : list
            Layer's output activations

        Returns
        -------
        torch.Tensor
            Filter's APoZ(average percentage of zeros) of the activations
        """
        activations = torch.cat(activations, 0)
        _eq_zero = torch.eq(activations, torch.zeros_like(activations))
        _apoz = torch.sum(_eq_zero, dim=(0, 2, 3)) / torch.numel(_eq_zero[:, 0, :, :])
        return _apoz


class ActivationMeanRankFilterPruner(ActivationRankFilterPruner):
    """
    A structured pruning algorithm that prunes the filters with the
    smallest mean value of output activations.
    Pavlo Molchanov, Stephen Tyree, Tero Karras, Timo Aila and Jan Kautz,
    "Pruning Convolutional Neural Networks for Resource Efficient Inference", ICLR 2017.
    https://arxiv.org/abs/1611.06440
    """

    def __init__(self, model, config_list, optimizer=None, activation='relu', statistics_batch_num=1):
        """
        Parameters
        ----------
        model : torch.nn.module
            Model to be pruned
        config_list : list
            support key for each list item:
                - sparsity: percentage of convolutional filters to be pruned.
        optimizer: torch.optim.Optimizer
            Optimizer used to train model
        activation : str
            Activation function
        statistics_batch_num : int
            Num of batches for activation statistics
        """
        super().__init__(model, config_list, optimizer, activation, statistics_batch_num)

    def get_mask(self, base_mask, activations, num_prune):
        """
        Calculate the mask of given layer.
        Filters with the smallest APoZ(average percentage of zeros) of output activations are masked.

        Parameters
        ----------
        base_mask : dict
            The basic mask with the same shape of weight, all item in the basic mask is 1.
        activations : list
            Layer's output activations
        num_prune : int
            Num of filters to prune

        Returns
        -------
        dict
            dictionary for storing masks
        """
        mean_activation = self._cal_mean_activation(activations)
        prune_indices = torch.argsort(mean_activation)[:num_prune]
        for idx in prune_indices:
            base_mask['weight_mask'][idx] = 0.
            if base_mask['bias_mask'] is not None:
                base_mask['bias_mask'][idx] = 0.
        return base_mask

    def _cal_mean_activation(self, activations):
        """
        Calculate mean value of activations.

        Parameters
        ----------
        activations : list
            Layer's output activations

        Returns
        -------
        torch.Tensor
            Filter's mean value of the output activations
        """
        activations = torch.cat(activations, 0)
        mean_activation = torch.mean(activations, dim=(0, 2, 3))
        return mean_activation
