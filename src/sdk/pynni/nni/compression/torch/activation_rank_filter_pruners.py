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

    def __init__(self, model, config_list, activation='relu', statistics_batch_num=1):
        """
        Parameters
        ----------
        model : torch.nn.module
            Model to be pruned
        config_list : list
            support key for each list item:
                - sparsity: percentage of convolutional filters to be pruned.
        activation : str
            Activation function
        statistics_batch_num : int
            Num of batches for activation statistics
        """

        super().__init__(model, config_list)
        self.register_buffer("if_calculated", torch.tensor(0)) # pylint: disable=not-callable
        self.statistics_batch_num = statistics_batch_num
        self.collected_activation = {}
        self.hooks = {}
        assert activation in ['relu', 'relu6']
        if activation == 'relu':
            self.activation = torch.nn.functional.relu
        elif activation == 'relu6':
            self.activation = torch.nn.functional.relu6
        else:
            self.activation = None

    def compress(self):
        """
        Compress the model, register a hook for collecting activations.
        """
        if self.modules_wrapper is not None:
            # already compressed
            return self.bound_model
        else:
            self.modules_wrapper = []
        modules_to_compress = self.detect_modules_to_compress()
        for layer, config in modules_to_compress:
            wrapper = self._wrap_modules(layer, config)
            self.modules_wrapper.append(wrapper)
            self.collected_activation[layer.name] = []

            def _hook(module_, input_, output, name=layer.name):
                if len(self.collected_activation[name]) < self.statistics_batch_num:
                    self.collected_activation[name].append(self.activation(output.detach().cpu()))

            wrapper.module.register_forward_hook(_hook)
        self._wrap_model()
        return self.bound_model

    def get_mask(self, base_mask, activations, num_prune):
        raise NotImplementedError('{} get_mask is not implemented'.format(self.__class__.__name__))

    def calc_mask(self, layer, config, **kwargs):
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

        weight = layer.module.weight.data
        op_type = layer.type
        assert 0 <= config.get('sparsity') < 1, "sparsity must in the range [0, 1)"
        assert op_type in ['Conv2d'], "only support Conv2d"
        assert op_type in config.get('op_types')
        if_calculated = kwargs["if_calculated"]
        if if_calculated:
            return None
        mask_weight = torch.ones(weight.size()).type_as(weight).detach()
        if hasattr(layer.module, 'bias') and layer.module.bias is not None:
            mask_bias = torch.ones(layer.module.bias.size()).type_as(layer.module.bias).detach()
        else:
            mask_bias = None
        mask = {'weight': mask_weight, 'bias': mask_bias}
        try:
            filters = weight.size(0)
            num_prune = int(filters * config.get('sparsity'))
            if filters < 2 or num_prune < 1 or len(self.collected_activation[layer.name]) < self.statistics_batch_num:
                return mask
            mask = self.get_mask(mask, self.collected_activation[layer.name], num_prune)
        finally:
            if len(self.collected_activation[layer.name]) == self.statistics_batch_num:
                if_calculated.copy_(torch.tensor(1)) # pylint: disable=not-callable
        return mask


class ActivationAPoZRankFilterPruner(ActivationRankFilterPruner):
    """
    A structured pruning algorithm that prunes the filters with the
    smallest APoZ(average percentage of zeros) of output activations.
    Hengyuan Hu, Rui Peng, Yu-Wing Tai and Chi-Keung Tang,
    "Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient Deep Architectures", ICLR 2016.
    https://arxiv.org/abs/1607.03250
    """

    def __init__(self, model, config_list, activation='relu', statistics_batch_num=1):
        """
        Parameters
        ----------
        model : torch.nn.module
            Model to be pruned
        config_list : list
            support key for each list item:
                - sparsity: percentage of convolutional filters to be pruned.
        activation : str
            Activation function
        statistics_batch_num : int
            Num of batches for activation statistics
        """
        super().__init__(model, config_list, activation, statistics_batch_num)

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
            base_mask['weight'][idx] = 0.
            if base_mask['bias'] is not None:
                base_mask['bias'][idx] = 0.
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

    def __init__(self, model, config_list, activation='relu', statistics_batch_num=1):
        """
        Parameters
        ----------
        model : torch.nn.module
            Model to be pruned
        config_list : list
            support key for each list item:
                - sparsity: percentage of convolutional filters to be pruned.
        activation : str
            Activation function
        statistics_batch_num : int
            Num of batches for activation statistics
        """
        super().__init__(model, config_list, activation, statistics_batch_num)

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
            base_mask['weight'][idx] = 0.
            if base_mask['bias'] is not None:
                base_mask['bias'][idx] = 0.
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
