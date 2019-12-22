# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import torch
from .compressor import Pruner

__all__ = ['LevelPruner', 'AGP_Pruner', 'SlimPruner', 'L1FilterPruner', 'L2FilterPruner', 'FPGMPruner',
           'ActivationAPoZRankFilterPruner', 'ActivationMeanRankFilterPruner']

logger = logging.getLogger('torch pruner')


class LevelPruner(Pruner):
    """
    Prune to an exact pruning level specification
    """

    def __init__(self, model, config_list):
        """
        Parameters
        ----------
        model : torch.nn.module
            Model to be pruned
        config_list : list
            List on pruning configs
        """

        super().__init__(model, config_list)
        self.mask_calculated_ops = set()

    def calc_mask(self, layer, config):
        """
        Calculate the mask of given layer
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
        op_name = layer.name
        if op_name not in self.mask_calculated_ops:
            w_abs = weight.abs()
            k = int(weight.numel() * config['sparsity'])
            if k == 0:
                return torch.ones(weight.shape).type_as(weight)
            threshold = torch.topk(w_abs.view(-1), k, largest=False)[0].max()
            mask_weight = torch.gt(w_abs, threshold).type_as(weight)
            mask = {'weight': mask_weight}
            self.mask_dict.update({op_name: mask})
            self.mask_calculated_ops.add(op_name)
        else:
            assert op_name in self.mask_dict
            mask = self.mask_dict[op_name]
        return mask


class AGP_Pruner(Pruner):
    """
    An automated gradual pruning algorithm that prunes the smallest magnitude
    weights to achieve a preset level of network sparsity.
    Michael Zhu and Suyog Gupta, "To prune, or not to prune: exploring the
    efficacy of pruning for model compression", 2017 NIPS Workshop on Machine
    Learning of Phones and other Consumer Devices,
    https://arxiv.org/pdf/1710.01878.pdf
    """

    def __init__(self, model, config_list):
        """
        Parameters
        ----------
        model : torch.nn.module
            Model to be pruned
        config_list : list
            List on pruning configs
        """

        super().__init__(model, config_list)
        self.now_epoch = 0
        self.if_init_list = {}

    def calc_mask(self, layer, config):
        """
        Calculate the mask of given layer
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
        op_name = layer.name
        start_epoch = config.get('start_epoch', 0)
        freq = config.get('frequency', 1)
        if self.now_epoch >= start_epoch and self.if_init_list.get(op_name, True) \
                and (self.now_epoch - start_epoch) % freq == 0:
            mask = self.mask_dict.get(op_name, {'weight': torch.ones(weight.shape).type_as(weight)})
            target_sparsity = self.compute_target_sparsity(config)
            k = int(weight.numel() * target_sparsity)
            if k == 0 or target_sparsity >= 1 or target_sparsity <= 0:
                return mask
            # if we want to generate new mask, we should update weigth first
            w_abs = weight.abs() * mask
            threshold = torch.topk(w_abs.view(-1), k, largest=False)[0].max()
            new_mask = {'weight': torch.gt(w_abs, threshold).type_as(weight)}
            self.mask_dict.update({op_name: new_mask})
            self.if_init_list.update({op_name: False})
        else:
            new_mask = self.mask_dict.get(op_name, {'weight': torch.ones(weight.shape).type_as(weight)})
        return new_mask

    def compute_target_sparsity(self, config):
        """
        Calculate the sparsity for pruning
        Parameters
        ----------
        config : dict
            Layer's pruning config
        Returns
        -------
        float
            Target sparsity to be pruned
        """

        end_epoch = config.get('end_epoch', 1)
        start_epoch = config.get('start_epoch', 0)
        freq = config.get('frequency', 1)
        final_sparsity = config.get('final_sparsity', 0)
        initial_sparsity = config.get('initial_sparsity', 0)
        if end_epoch <= start_epoch or initial_sparsity >= final_sparsity:
            logger.warning('your end epoch <= start epoch or initial_sparsity >= final_sparsity')
            return final_sparsity

        if end_epoch <= self.now_epoch:
            return final_sparsity

        span = ((end_epoch - start_epoch - 1) // freq) * freq
        assert span > 0
        target_sparsity = (final_sparsity +
                           (initial_sparsity - final_sparsity) *
                           (1.0 - ((self.now_epoch - start_epoch) / span)) ** 3)
        return target_sparsity

    def update_epoch(self, epoch):
        """
        Update epoch
        Parameters
        ----------
        epoch : int
            current training epoch
        """

        if epoch > 0:
            self.now_epoch = epoch
            for k in self.if_init_list.keys():
                self.if_init_list[k] = True


class SlimPruner(Pruner):
    """
    A structured pruning algorithm that prunes channels by pruning the weights of BN layers.
    Zhuang Liu, Jianguo Li, Zhiqiang Shen, Gao Huang, Shoumeng Yan and Changshui Zhang
    "Learning Efficient Convolutional Networks through Network Slimming", 2017 ICCV
    https://arxiv.org/pdf/1708.06519.pdf
    """

    def __init__(self, model, config_list):
        """
        Parameters
        ----------
        config_list : list
            support key for each list item:
                - sparsity: percentage of convolutional filters to be pruned.
        """

        super().__init__(model, config_list)
        self.mask_calculated_ops = set()
        weight_list = []
        if len(config_list) > 1:
            logger.warning('Slim pruner only supports 1 configuration')
        config = config_list[0]
        for (layer, config) in self.detect_modules_to_compress():
            assert layer.type == 'BatchNorm2d', 'SlimPruner only supports 2d batch normalization layer pruning'
            weight_list.append(layer.module.weight.data.abs().clone())
        all_bn_weights = torch.cat(weight_list)
        k = int(all_bn_weights.shape[0] * config['sparsity'])
        self.global_threshold = torch.topk(all_bn_weights.view(-1), k, largest=False)[0].max()

    def calc_mask(self, layer, config):
        """
        Calculate the mask of given layer.
        Scale factors with the smallest absolute value in the BN layer are masked.
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
        op_name = layer.name
        op_type = layer.type
        assert op_type == 'BatchNorm2d', 'SlimPruner only supports 2d batch normalization layer pruning'
        if op_name in self.mask_calculated_ops:
            assert op_name in self.mask_dict
            return self.mask_dict.get(op_name)
        base_mask = torch.ones(weight.size()).type_as(weight).detach()
        mask = {'weight': base_mask.detach(), 'bias': base_mask.clone().detach()}
        try:
            filters = weight.size(0)
            num_prune = int(filters * config.get('sparsity'))
            if filters < 2 or num_prune < 1:
                return mask
            w_abs = weight.abs()
            mask_weight = torch.gt(w_abs, self.global_threshold).type_as(weight)
            mask_bias = mask_weight.clone()
            mask = {'weight': mask_weight.detach(), 'bias': mask_bias.detach()}
        finally:
            self.mask_dict.update({layer.name: mask})
            self.mask_calculated_ops.add(layer.name)

        return mask


class WeightRankFilterPruner(Pruner):
    """
    A structured pruning base class that prunes the filters with the smallest
    importance criterion in convolution layers to achieve a preset level of network sparsity.
    """

    def __init__(self, model, config_list):
        """
        Parameters
        ----------
        model : torch.nn.module
            Model to be pruned
        config_list : list
            support key for each list item:
                - sparsity: percentage of convolutional filters to be pruned.
        """

        super().__init__(model, config_list)
        self.mask_calculated_ops = set()

    def _get_mask(self, base_mask, weight, num_prune):
        return {'weight': None, 'bias': None}

    def calc_mask(self, layer, config):
        """
        Calculate the mask of given layer.
        Filters with the smallest importance criterion of the kernel weights are masked.
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
        op_name = layer.name
        op_type = layer.type
        assert 0 <= config.get('sparsity') < 1
        assert op_type in ['Conv1d', 'Conv2d']
        assert op_type in config.get('op_types')
        if op_name in self.mask_calculated_ops:
            assert op_name in self.mask_dict
            return self.mask_dict.get(op_name)
        mask_weight = torch.ones(weight.size()).type_as(weight).detach()
        if hasattr(layer.module, 'bias') and layer.module.bias is not None:
            mask_bias = torch.ones(layer.module.bias.size()).type_as(layer.module.bias).detach()
        else:
            mask_bias = None
        mask = {'weight': mask_weight, 'bias': mask_bias}
        try:
            filters = weight.size(0)
            num_prune = int(filters * config.get('sparsity'))
            if filters < 2 or num_prune < 1:
                return mask
            mask = self._get_mask(mask, weight, num_prune)
        finally:
            self.mask_dict.update({op_name: mask})
            self.mask_calculated_ops.add(op_name)
        return mask


class L1FilterPruner(WeightRankFilterPruner):
    """
    A structured pruning algorithm that prunes the filters of smallest magnitude
    weights sum in the convolution layers to achieve a preset level of network sparsity.
    Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet and Hans Peter Graf,
    "PRUNING FILTERS FOR EFFICIENT CONVNETS", 2017 ICLR
    https://arxiv.org/abs/1608.08710
    """

    def __init__(self, model, config_list):
        """
        Parameters
        ----------
        model : torch.nn.module
            Model to be pruned
        config_list : list
            support key for each list item:
                - sparsity: percentage of convolutional filters to be pruned.
        """

        super().__init__(model, config_list)

    def _get_mask(self, base_mask, weight, num_prune):
        """
        Calculate the mask of given layer.
        Filters with the smallest sum of its absolute kernel weights are masked.
        Parameters
        ----------
        base_mask : dict
            The basic mask with the same shape of weight or bias, all item in the basic mask is 1.
        weight : torch.Tensor
            Layer's weight
        num_prune : int
            Num of filters to prune
        Returns
        -------
        dict
            dictionary for storing masks
        """

        filters = weight.shape[0]
        w_abs = weight.abs()
        w_abs_structured = w_abs.view(filters, -1).sum(dim=1)
        threshold = torch.topk(w_abs_structured.view(-1), num_prune, largest=False)[0].max()
        mask_weight = torch.gt(w_abs_structured, threshold)[:, None, None, None].expand_as(weight).type_as(weight)
        mask_bias = torch.gt(w_abs_structured, threshold).type_as(weight)

        return {'weight': mask_weight.detach(), 'bias': mask_bias.detach()}


class L2FilterPruner(WeightRankFilterPruner):
    """
    A structured pruning algorithm that prunes the filters with the
    smallest L2 norm of the weights.
    """

    def __init__(self, model, config_list):
        """
        Parameters
        ----------
        model : torch.nn.module
            Model to be pruned
        config_list : list
            support key for each list item:
                - sparsity: percentage of convolutional filters to be pruned.
        """

        super().__init__(model, config_list)

    def _get_mask(self, base_mask, weight, num_prune):
        """
        Calculate the mask of given layer.
        Filters with the smallest L2 norm of the absolute kernel weights are masked.
        Parameters
        ----------
        base_mask : dict
            The basic mask with the same shape of weight or bias, all item in the basic mask is 1.
        weight : torch.Tensor
            Layer's weight
        num_prune : int
            Num of filters to prune
        Returns
        -------
        dict
            dictionary for storing masks
        """
        filters = weight.shape[0]
        w = weight.view(filters, -1)
        w_l2_norm = torch.sqrt((w ** 2).sum(dim=1))
        threshold = torch.topk(w_l2_norm.view(-1), num_prune, largest=False)[0].max()
        mask_weight = torch.gt(w_l2_norm, threshold)[:, None, None, None].expand_as(weight).type_as(weight)
        mask_bias = torch.gt(w_l2_norm, threshold).type_as(weight)

        return {'weight': mask_weight.detach(), 'bias': mask_bias.detach()}


class FPGMPruner(WeightRankFilterPruner):
    """
    A filter pruner via geometric median.
    "Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration",
    https://arxiv.org/pdf/1811.00250.pdf
    """

    def __init__(self, model, config_list):
        """
        Parameters
        ----------
        model : pytorch model
            the model user wants to compress
        config_list: list
            support key for each list item:
                - sparsity: percentage of convolutional filters to be pruned.
        """
        super().__init__(model, config_list)

    def _get_mask(self, base_mask, weight, num_prune):
        """
        Calculate the mask of given layer.
        Filters with the smallest sum of its absolute kernel weights are masked.
        Parameters
        ----------
        base_mask : dict
            The basic mask with the same shape of weight and bias, all item in the basic mask is 1.
        weight : torch.Tensor
            Layer's weight
        num_prune : int
            Num of filters to prune
        Returns
        -------
        dict
            dictionary for storing masks
        """
        min_gm_idx = self._get_min_gm_kernel_idx(weight, num_prune)
        for idx in min_gm_idx:
            base_mask['weight'][idx] = 0.
            if base_mask['bias'] is not None:
                base_mask['bias'][idx] = 0.
        return base_mask

    def _get_min_gm_kernel_idx(self, weight, n):
        assert len(weight.size()) in [3, 4]

        dist_list = []
        for out_i in range(weight.size(0)):
            dist_sum = self._get_distance_sum(weight, out_i)
            dist_list.append((dist_sum, out_i))
        min_gm_kernels = sorted(dist_list, key=lambda x: x[0])[:n]
        return [x[1] for x in min_gm_kernels]

    def _get_distance_sum(self, weight, out_idx):
        """
        Calculate the total distance between a specified filter (by out_idex and in_idx) and
        all other filters.
        Optimized verision of following naive implementation:
        def _get_distance_sum(self, weight, in_idx, out_idx):
            w = weight.view(-1, weight.size(-2), weight.size(-1))
            dist_sum = 0.
            for k in w:
                dist_sum += torch.dist(k, weight[in_idx, out_idx], p=2)
            return dist_sum
        Parameters
        ----------
        weight: Tensor
            convolutional filter weight
        out_idx: int
            output channel index of specified filter, this method calculates the total distance
            between this specified filter and all other filters.
        Returns
        -------
        float32
            The total distance
        """
        logger.debug('weight size: %s', weight.size())
        assert len(weight.size()) in [3, 4], 'unsupported weight shape'

        w = weight.view(weight.size(0), -1)
        anchor_w = w[out_idx].unsqueeze(0).expand(w.size(0), w.size(1))
        x = w - anchor_w
        x = (x * x).sum(-1)
        x = torch.sqrt(x)
        return x.sum()

    def update_epoch(self, epoch):
        self.mask_calculated_ops = set()


class ActivationRankFilterPruner(Pruner):
    """
    A structured pruning base class that prunes the filters with the smallest
    importance criterion in convolution layers to achieve a preset level of network sparsity.
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
        self.mask_calculated_ops = set()
        self.statistics_batch_num = statistics_batch_num
        self.collected_activation = {}
        self.hooks = {}
        assert activation in ['relu']
        if activation == 'relu':
            self.activation = torch.nn.functional.relu
        else:
            self.activation = None

    def compress(self):
        """
        Compress the model, register a hook for collecting activations.
        """
        modules_to_compress = self.detect_modules_to_compress()
        for layer, config in modules_to_compress:
            self._instrument_layer(layer, config)
            self.collected_activation[layer.name] = []

            def _hook(module_, input_, output, name=layer.name):
                if len(self.collected_activation[name]) < self.statistics_batch_num:
                    self.collected_activation[name].append(self.activation(output.detach().cpu()))

            layer.module.register_forward_hook(_hook)
        return self.bound_model

    def _get_mask(self, base_mask, activations, num_prune):
        return {'weight': None, 'bias': None}

    def calc_mask(self, layer, config):
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
        op_name = layer.name
        op_type = layer.type
        assert 0 <= config.get('sparsity') < 1
        assert op_type in ['Conv2d']
        assert op_type in config.get('op_types')
        if op_name in self.mask_calculated_ops:
            assert op_name in self.mask_dict
            return self.mask_dict.get(op_name)
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
            mask = self._get_mask(mask, self.collected_activation[layer.name], num_prune)
        finally:
            if len(self.collected_activation[layer.name]) == self.statistics_batch_num:
                self.mask_dict.update({op_name: mask})
                self.mask_calculated_ops.add(op_name)
        return mask


class ActivationAPoZRankFilterPruner(ActivationRankFilterPruner):
    """
    A structured pruning algorithm that prunes the filters with the
    smallest APoZ(average percentage of zeros) of output activations.
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

    def _get_mask(self, base_mask, activations, num_prune):
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
        prune_indices = torch.argsort(apoz)[:num_prune]
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

    def _get_mask(self, base_mask, activations, num_prune):
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
