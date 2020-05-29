# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import torch
from .pruners import WeightMasker

__all__ = ['L1FilterPrunerMasker', 'L2FilterPrunerMasker', 'FPGMPrunerMasker', 'TaylorFOWeightFilterPrunerMasker']

logger = logging.getLogger('torch weight rank filter pruners')

class WeightRankMasker(WeightMasker):
    """
    A structured pruning base class that prunes the filters with the smallest
    importance criterion in convolution layers to achieve a preset level of network sparsity.
    """
    def calc_mask(self, weight, bias=None, sparsity=1., wrapper=None):
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

        mask_weight = torch.ones(weight.size()).type_as(weight).detach()
        if bias is not None:
            mask_bias = torch.ones(bias.size()).type_as(bias).detach()
        else:
            mask_bias = None
        mask = {'weight_mask': mask_weight, 'bias_mask': mask_bias}

        filters = weight.size(0)
        num_prune = int(filters * sparsity)
        if filters < 2 or num_prune < 1:
            return mask

        return self.get_mask(mask, weight, num_prune, wrapper)

    def get_mask(self, base_mask, weight, num_prune, wrapper):
        """
        Calculate the mask of given layer.
        Filters with the smallest importance approximations are masked.

        Parameters
        ----------
        base_mask: dict
            The basic mask with the same shape of weight, all item in the basic mask is 1.
        weight: tensor
            the module weight to be pruned
        num_prune: int
            Num of filters to prune
        wrapper:
            the wrapper object of the layer to be pruned

        Returns
        -------
        dict
            dictionary for storing masks
        """
        raise NotImplementedError('{} get_mask is not implemented'.format(self.__class__.__name__))

class L1FilterPrunerMasker(WeightRankMasker):
    """
    A structured pruning algorithm that prunes the filters of smallest magnitude
    weights sum in the convolution layers to achieve a preset level of network sparsity.
    Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet and Hans Peter Graf,
    "PRUNING FILTERS FOR EFFICIENT CONVNETS", 2017 ICLR
    https://arxiv.org/abs/1608.08710
    """

    def get_mask(self, base_mask, weight, num_prune, wrapper):
        filters = weight.shape[0]
        w_abs = weight.abs()
        w_abs_structured = w_abs.view(filters, -1).sum(dim=1)
        threshold = torch.topk(w_abs_structured.view(-1), num_prune, largest=False)[0].max()
        mask_weight = torch.gt(w_abs_structured, threshold)[:, None, None, None].expand_as(weight).type_as(weight)
        mask_bias = torch.gt(w_abs_structured, threshold).type_as(weight).detach() if base_mask['bias_mask'] is not None else None

        return {'weight_mask': mask_weight.detach(), 'bias_mask': mask_bias}

class L2FilterPrunerMasker(WeightRankMasker):
    """
    A structured pruning algorithm that prunes the filters with the
    smallest L2 norm of the weights.
    """
    def get_mask(self, base_mask, weight, num_prune, wrapper):
        filters = weight.shape[0]
        w = weight.view(filters, -1)
        w_l2_norm = torch.sqrt((w ** 2).sum(dim=1))
        threshold = torch.topk(w_l2_norm.view(-1), num_prune, largest=False)[0].max()
        mask_weight = torch.gt(w_l2_norm, threshold)[:, None, None, None].expand_as(weight).type_as(weight)
        mask_bias = torch.gt(w_l2_norm, threshold).type_as(weight).detach() if base_mask['bias_mask'] is not None else None

        return {'weight_mask': mask_weight.detach(), 'bias_mask': mask_bias}


class FPGMPrunerMasker(WeightRankMasker):
    """
    A filter pruner via geometric median.
    "Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration",
    https://arxiv.org/pdf/1811.00250.pdf
    """
    def get_mask(self, base_mask, weight, num_prune, wrapper):
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

class TaylorFOWeightFilterPrunerMasker(WeightRankMasker):
    """
    A structured pruning algorithm that prunes the filters with the smallest
    importance approximations based on the first order taylor expansion on the weight.
    Molchanov, Pavlo and Mallya, Arun and Tyree, Stephen and Frosio, Iuri and Kautz, Jan,
    "Importance Estimation for Neural Network Pruning", CVPR 2019.
    http://jankautz.com/publications/Importance4NNPruning_CVPR19.pdf
    """

    def get_mask(self, base_mask, weight, num_prune, wrapper):
        prune_indices = torch.argsort(wrapper.contribution)[:num_prune]
        for idx in prune_indices:
            base_mask['weight_mask'][idx] = 0.
            if base_mask['bias_mask'] is not None:
                base_mask['bias_mask'][idx] = 0.
        return base_mask
