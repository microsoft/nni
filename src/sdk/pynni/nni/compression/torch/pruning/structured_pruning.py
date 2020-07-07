# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import torch
from .weight_masker import WeightMasker

__all__ = ['L1FilterPrunerMasker', 'L2FilterPrunerMasker', 'FPGMPrunerMasker', \
    'TaylorFOWeightFilterPrunerMasker', 'ActivationAPoZRankFilterPrunerMasker', \
    'ActivationMeanRankFilterPrunerMasker', 'SlimPrunerMasker']

logger = logging.getLogger('torch filter pruners')

class StructuredWeightMasker(WeightMasker):
    """
    A structured pruning masker base class that prunes convolutional layer filters.
    """
    def calc_mask(self, sparsity, wrapper, wrapper_idx=None):
        """
        Calculate the mask of given layer.
        Parameters
        ----------
        sparsity: float
            pruning ratio,  preserved weight ratio is `1 - sparsity`
        wrapper: PrunerModuleWrapper
            layer wrapper of this layer
        wrapper_idx: int
            index of this wrapper in pruner's all wrappers
        Returns
        -------
        dict
            dictionary for storing masks, keys of the dict:
            'weight_mask':  weight mask tensor
            'bias_mask': bias mask tensor (optional)
        """
        msg = 'module type {} is not supported!'.format(wrapper.type)
        assert wrapper.type == 'Conv2d', msg
        weight = wrapper.module.weight.data
        bias = None
        if hasattr(wrapper.module, 'bias') and wrapper.module.bias is not None:
            bias = wrapper.module.bias.data

        if wrapper.weight_mask is None:
            mask_weight = torch.ones(weight.size()).type_as(weight).detach()
        else:
            mask_weight = wrapper.weight_mask.clone()
        if bias is not None:
            if wrapper.bias_mask is None:
                mask_bias = torch.ones(bias.size()).type_as(bias).detach()
            else:
                mask_bias = wrapper.bias_mask.clone()
        else:
            mask_bias = None
        mask = {'weight_mask': mask_weight, 'bias_mask': mask_bias}

        filters = weight.size(0)
        num_prune = int(filters * sparsity)
        if filters < 2 or num_prune < 1:
            return mask
        # weight*mask_weight: apply base mask for iterative pruning
        return self.get_mask(mask, weight*mask_weight, num_prune, wrapper, wrapper_idx)

    def get_mask(self, base_mask, weight, num_prune, wrapper, wrapper_idx):
        """
        Calculate the mask of given layer.
        Parameters
        ----------
        base_mask: dict
            The basic mask with the same shape of weight, all item in the basic mask is 1.
        weight: tensor
            the module weight to be pruned
        num_prune: int
            Num of filters to prune
        wrapper: PrunerModuleWrapper
            layer wrapper of this layer
        wrapper_idx: int
            index of this wrapper in pruner's all wrappers
        Returns
        -------
        dict
            dictionary for storing masks
        """
        raise NotImplementedError('{} get_mask is not implemented'.format(self.__class__.__name__))

class L1FilterPrunerMasker(StructuredWeightMasker):
    """
    A structured pruning algorithm that prunes the filters of smallest magnitude
    weights sum in the convolution layers to achieve a preset level of network sparsity.
    Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet and Hans Peter Graf,
    "PRUNING FILTERS FOR EFFICIENT CONVNETS", 2017 ICLR
    https://arxiv.org/abs/1608.08710
    """

    def get_mask(self, base_mask, weight, num_prune, wrapper, wrapper_idx):
        filters = weight.shape[0]
        w_abs = weight.abs()
        w_abs_structured = w_abs.view(filters, -1).sum(dim=1)
        threshold = torch.topk(w_abs_structured.view(-1), num_prune, largest=False)[0].max()
        mask_weight = torch.gt(w_abs_structured, threshold)[:, None, None, None].expand_as(weight).type_as(weight)
        mask_bias = torch.gt(w_abs_structured, threshold).type_as(weight).detach() if base_mask['bias_mask'] is not None else None

        return {'weight_mask': mask_weight.detach(), 'bias_mask': mask_bias}

class L2FilterPrunerMasker(StructuredWeightMasker):
    """
    A structured pruning algorithm that prunes the filters with the
    smallest L2 norm of the weights.
    """
    def get_mask(self, base_mask, weight, num_prune, wrapper, wrapper_idx):
        filters = weight.shape[0]
        w = weight.view(filters, -1)
        w_l2_norm = torch.sqrt((w ** 2).sum(dim=1))
        threshold = torch.topk(w_l2_norm.view(-1), num_prune, largest=False)[0].max()
        mask_weight = torch.gt(w_l2_norm, threshold)[:, None, None, None].expand_as(weight).type_as(weight)
        mask_bias = torch.gt(w_l2_norm, threshold).type_as(weight).detach() if base_mask['bias_mask'] is not None else None

        return {'weight_mask': mask_weight.detach(), 'bias_mask': mask_bias}


class FPGMPrunerMasker(StructuredWeightMasker):
    """
    A filter pruner via geometric median.
    "Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration",
    https://arxiv.org/pdf/1811.00250.pdf
    """
    def get_mask(self, base_mask, weight, num_prune, wrapper, wrapper_idx):
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

class TaylorFOWeightFilterPrunerMasker(StructuredWeightMasker):
    """
    A structured pruning algorithm that prunes the filters with the smallest
    importance approximations based on the first order taylor expansion on the weight.
    Molchanov, Pavlo and Mallya, Arun and Tyree, Stephen and Frosio, Iuri and Kautz, Jan,
    "Importance Estimation for Neural Network Pruning", CVPR 2019.
    http://jankautz.com/publications/Importance4NNPruning_CVPR19.pdf
    """
    def __init__(self, model, pruner, statistics_batch_num=1):
        super().__init__(model, pruner)
        self.pruner.statistics_batch_num = statistics_batch_num
        self.pruner.set_wrappers_attribute("contribution", None)
        self.pruner.iterations = 0
        self.pruner.patch_optimizer(self.calc_contributions)

    def get_mask(self, base_mask, weight, num_prune, wrapper, wrapper_idx):
        if self.pruner.iterations < self.pruner.statistics_batch_num:
            return None

        if wrapper.contribution is None:
            return None

        prune_indices = torch.argsort(wrapper.contribution)[:num_prune]
        for idx in prune_indices:
            base_mask['weight_mask'][idx] = 0.
            if base_mask['bias_mask'] is not None:
                base_mask['bias_mask'][idx] = 0.
        return base_mask

    def calc_contributions(self):
        """
        Calculate the estimated importance of filters as a sum of individual contribution
        based on the first order taylor expansion.
        """
        if self.pruner.iterations >= self.pruner.statistics_batch_num:
            return
        for wrapper in self.pruner.get_modules_wrapper():
            filters = wrapper.module.weight.size(0)
            contribution = (wrapper.module.weight*wrapper.module.weight.grad).data.pow(2).view(filters, -1).sum(dim=1)
            if wrapper.contribution is None:
                wrapper.contribution = contribution
            else:
                wrapper.contribution += contribution

        self.pruner.iterations += 1


class ActivationFilterPrunerMasker(StructuredWeightMasker):
    def __init__(self, model, pruner, statistics_batch_num=1, activation='relu'):
        super().__init__(model, pruner)
        self.statistics_batch_num = statistics_batch_num
        self.pruner.hook_id = self._add_activation_collector(self.pruner)

        assert activation in ['relu', 'relu6']
        if activation == 'relu':
            self.pruner.activation = torch.nn.functional.relu
        elif activation == 'relu6':
            self.pruner.activation = torch.nn.functional.relu6
        else:
            self.pruner.activation = None

    def _add_activation_collector(self, pruner):
        def collector(collected_activation):
            def hook(module_, input_, output):
                collected_activation.append(pruner.activation(output.detach().cpu()))
            return hook
        pruner.collected_activation = {}
        pruner._fwd_hook_id += 1
        pruner._fwd_hook_handles[pruner._fwd_hook_id] = []

        for wrapper_idx, wrapper in enumerate(pruner.get_modules_wrapper()):
            pruner.collected_activation[wrapper_idx] = []
            handle = wrapper.register_forward_hook(collector(pruner.collected_activation[wrapper_idx]))

            pruner._fwd_hook_handles[pruner._fwd_hook_id].append(handle)
        return pruner._fwd_hook_id

class ActivationAPoZRankFilterPrunerMasker(ActivationFilterPrunerMasker):
    """
    A structured pruning algorithm that prunes the filters with the
    smallest APoZ(average percentage of zeros) of output activations.
    Hengyuan Hu, Rui Peng, Yu-Wing Tai and Chi-Keung Tang,
    "Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient Deep Architectures", ICLR 2016.
    https://arxiv.org/abs/1607.03250
    """
    def get_mask(self, base_mask, weight, num_prune, wrapper, wrapper_idx):
        assert wrapper_idx is not None
        activations = self.pruner.collected_activation[wrapper_idx]
        if len(activations) < self.statistics_batch_num:
            return None
        apoz = self._calc_apoz(activations)
        prune_indices = torch.argsort(apoz, descending=True)[:num_prune]
        for idx in prune_indices:
            base_mask['weight_mask'][idx] = 0.
            if base_mask['bias_mask'] is not None:
                base_mask['bias_mask'][idx] = 0.

        if len(activations) >= self.statistics_batch_num and self.pruner.hook_id in self.pruner._fwd_hook_handles:
            self.pruner.remove_activation_collector(self.pruner.hook_id)

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

class ActivationMeanRankFilterPrunerMasker(ActivationFilterPrunerMasker):
    """
    A structured pruning algorithm that prunes the filters with the
    smallest mean value of output activations.
    Pavlo Molchanov, Stephen Tyree, Tero Karras, Timo Aila and Jan Kautz,
    "Pruning Convolutional Neural Networks for Resource Efficient Inference", ICLR 2017.
    https://arxiv.org/abs/1611.06440
    """
    def get_mask(self, base_mask, weight, num_prune, wrapper, wrapper_idx):
        assert wrapper_idx is not None
        activations = self.pruner.collected_activation[wrapper_idx]
        if len(activations) < self.statistics_batch_num:
            return None
        mean_activation = self._cal_mean_activation(activations)
        prune_indices = torch.argsort(mean_activation)[:num_prune]
        for idx in prune_indices:
            base_mask['weight_mask'][idx] = 0.
            if base_mask['bias_mask'] is not None:
                base_mask['bias_mask'][idx] = 0.

        if len(activations) >= self.statistics_batch_num and self.pruner.hook_id in self.pruner._fwd_hook_handles:
            self.pruner.remove_activation_collector(self.pruner.hook_id)

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

class SlimPrunerMasker(WeightMasker):
    """
    A structured pruning algorithm that prunes channels by pruning the weights of BN layers.
    Zhuang Liu, Jianguo Li, Zhiqiang Shen, Gao Huang, Shoumeng Yan and Changshui Zhang
    "Learning Efficient Convolutional Networks through Network Slimming", 2017 ICCV
    https://arxiv.org/pdf/1708.06519.pdf
    """

    def __init__(self, model, pruner, **kwargs):
        super().__init__(model, pruner)
        weight_list = []
        for (layer, _) in pruner.get_modules_to_compress():
            weight_list.append(layer.module.weight.data.abs().clone())
        all_bn_weights = torch.cat(weight_list)
        k = int(all_bn_weights.shape[0] * pruner.config_list[0]['sparsity'])
        self.global_threshold = torch.topk(all_bn_weights.view(-1), k, largest=False)[0].max()

    def calc_mask(self, sparsity, wrapper, wrapper_idx=None):
        assert wrapper.type == 'BatchNorm2d', 'SlimPruner only supports 2d batch normalization layer pruning'
        weight = wrapper.module.weight.data.clone()
        if wrapper.weight_mask is not None:
            # apply base mask for iterative pruning
            weight = weight * wrapper.weight_mask

        base_mask = torch.ones(weight.size()).type_as(weight).detach()
        mask = {'weight_mask': base_mask.detach(), 'bias_mask': base_mask.clone().detach()}
        filters = weight.size(0)
        num_prune = int(filters * sparsity)
        if filters >= 2 and num_prune >= 1:
            w_abs = weight.abs()
            mask_weight = torch.gt(w_abs, self.global_threshold).type_as(weight)
            mask_bias = mask_weight.clone()
            mask = {'weight_mask': mask_weight.detach(), 'bias_mask': mask_bias.detach()}
        return mask
