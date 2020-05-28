# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import torch
from schema import And, Optional
from ..utils.config_validation import CompressorSchema
from ..compressor import Pruner

__all__ = ['L1FilterPruner', 'L2FilterPruner', 'FPGMPruner']

logger = logging.getLogger('torch weight rank filter pruners')

class WeightRankFilterPruner(Pruner):
    """
    A structured pruning base class that prunes the filters with the smallest
    importance criterion in convolution layers to achieve a preset level of network sparsity.
    """

    def __init__(self, model, config_list, optimizer=None):
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
        """

        super().__init__(model, config_list, optimizer)
        self.set_wrappers_attribute("if_calculated", False)

    def validate_config(self, model, config_list):
        """
        Parameters
        ----------
        model : torch.nn.module
            Model to be pruned
        config_list : list
            support key for each list item:
                - sparsity: percentage of convolutional filters to be pruned.
       """
        schema = CompressorSchema([{
            'sparsity': And(float, lambda n: 0 < n < 1),
            Optional('op_types'): ['Conv2d'],
            Optional('op_names'): [str]
        }], model, logger)

        schema.validate(config_list)

    def get_mask(self, base_mask, weight, num_prune):
        raise NotImplementedError('{} get_mask is not implemented'.format(self.__class__.__name__))

    def calc_mask(self, wrapper, **kwargs):
        """
        Calculate the mask of given layer.
        Filters with the smallest importance criterion of the kernel weights are masked.
        Parameters
        ----------
        wrapper : Module
            the module to instrument the compression operation
        Returns
        -------
        dict
            dictionary for storing masks
        """

        weight = wrapper.module.weight.data
        op_type = wrapper.type
        config = wrapper.config
        assert 0 <= config.get('sparsity') < 1, "sparsity must in the range [0, 1)"
        assert op_type in ['Conv1d', 'Conv2d'], "only support Conv1d and Conv2d"
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
            if filters < 2 or num_prune < 1:
                return mask
            mask = self.get_mask(mask, weight, num_prune)
        finally:
            wrapper.if_calculated = True
        return mask


class L1FilterPruner(WeightRankFilterPruner):
    """
    A structured pruning algorithm that prunes the filters of smallest magnitude
    weights sum in the convolution layers to achieve a preset level of network sparsity.
    Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet and Hans Peter Graf,
    "PRUNING FILTERS FOR EFFICIENT CONVNETS", 2017 ICLR
    https://arxiv.org/abs/1608.08710
    """

    def __init__(self, model, config_list, optimizer=None):
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
        """

        super().__init__(model, config_list, optimizer)

    def get_mask(self, base_mask, weight, num_prune):
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
        mask_bias = torch.gt(w_abs_structured, threshold).type_as(weight).detach() if base_mask['bias_mask'] is not None else None

        return {'weight_mask': mask_weight.detach(), 'bias_mask': mask_bias}


class L2FilterPruner(WeightRankFilterPruner):
    """
    A structured pruning algorithm that prunes the filters with the
    smallest L2 norm of the weights.
    """

    def __init__(self, model, config_list, optimizer=None):
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
        """

        super().__init__(model, config_list, optimizer)

    def get_mask(self, base_mask, weight, num_prune):
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
        mask_bias = torch.gt(w_l2_norm, threshold).type_as(weight).detach() if base_mask['bias_mask'] is not None else None

        return {'weight_mask': mask_weight.detach(), 'bias_mask': mask_bias}


class FPGMPruner(WeightRankFilterPruner):
    """
    A filter pruner via geometric median.
    "Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration",
    https://arxiv.org/pdf/1811.00250.pdf
    """

    def __init__(self, model, config_list, optimizer):
        """
        Parameters
        ----------
        model : pytorch model
            the model user wants to compress
        config_list: list
            support key for each list item:
                - sparsity: percentage of convolutional filters to be pruned.
        optimizer: torch.optim.Optimizer
            Optimizer used to train model
        """
        super().__init__(model, config_list, optimizer)
        assert isinstance(optimizer, torch.optim.Optimizer), "FPGM pruner is an iterative pruner, please pass optimizer of the model to it"

    def get_mask(self, base_mask, weight, num_prune):
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
            base_mask['weight_mask'][idx] = 0.
            if base_mask['bias_mask'] is not None:
                base_mask['bias_mask'][idx] = 0.
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
        for wrapper in self.get_modules_wrapper():
            wrapper.if_calculated = False
