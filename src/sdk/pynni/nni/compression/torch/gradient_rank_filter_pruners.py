# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import torch
from .compressor import Pruner

__all__ = ['TaylorFOWeightFilterPruner', 'TaylorFOWeightFilterGlobalPruner']

logger = logging.getLogger('torch gradient rank filter pruners')

class GradientRankFilterPruner(Pruner):
    """
    A structured pruning base class that prunes the filters with the smallest
    importance criterion in convolution layers (using gradient values)
    to achieve a preset level of network sparsity.
    """

    def __init__(self, model, config_list, optimizer, statistics_batch_num=1):
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
        statistics_batch_num : int
            Num of batches for calculating contribution
        """

        super().__init__(model, config_list, optimizer)
        self.set_wrappers_attribute("if_calculated", False)
        self.set_wrappers_attribute("contribution", None)
        self.statistics_batch_num = statistics_batch_num
        self.iterations = 0
        self.patch_optimizer(self.calc_contributions)

    def calc_contributions(self):
        raise NotImplementedError('{} calc_contributions is not implemented'.format(self.__class__.__name__))

    # def get_mask(self, base_mask, contribution, num_prune):
    #     raise NotImplementedError('{} get_mask is not implemented'.format(self.__class__.__name__))

    def calc_mask(self, wrapper, **kwargs):
        """
        Calculate the mask of given layer.
        Filters with the smallest importance criterion which is calculated from the activation are masked.

        Parameters
        ----------
        wrapper : Module
            the layer to instrument the compression operation

        Returns
        -------
        dict
            dictionary for storing masks
        """

        weight = wrapper.module.weight.data
        op_type = wrapper.type
        config = wrapper.config
        assert 0 <= config.get('sparsity') < 1, "sparsity must in the range [0, 1)"
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
            if filters < 2 or num_prune < 1 or self.iterations < self.statistics_batch_num:
                return mask

            mask = self.get_mask(mask, wrapper.contribution, num_prune)
        finally:
            if self.iterations >= self.statistics_batch_num:
                wrapper.if_calculated = True

        return mask


class TaylorFOWeightFilterPruner(GradientRankFilterPruner):
    """
    A structured pruning algorithm that prunes the filters with the smallest
    importance approximations based on the first order taylor expansion on the weight.
    Molchanov, Pavlo and Mallya, Arun and Tyree, Stephen and Frosio, Iuri and Kautz, Jan,
    "Importance Estimation for Neural Network Pruning", CVPR 2019.
    http://jankautz.com/publications/Importance4NNPruning_CVPR19.pdf
    """

    def __init__(self, model, config_list, optimizer, statistics_batch_num=1):
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
        statistics_batch_num : int
            Num of batches for activation statistics
        """
        super().__init__(model, config_list, optimizer, statistics_batch_num)

    def get_mask(self, base_mask, contribution, num_prune):
        """
        Calculate the mask of given layer.
        Filters with the smallest importance approximations are masked.

        Parameters
        ----------
        base_mask : dict
            The basic mask with the same shape of weight, all item in the basic mask is 1.
        contribution : torch.Tensor
            Layer's importance approximations
        num_prune : int
            Num of filters to prune

        Returns
        -------
        dict
            dictionary for storing masks
        """
        prune_indices = torch.argsort(contribution)[:num_prune]
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
        if self.iterations >= self.statistics_batch_num:
            return
        for wrapper in self.get_modules_wrapper():
            filters = wrapper.module.weight.size(0)
            contribution = (wrapper.module.weight*wrapper.module.weight.grad).data.pow(2).view(filters, -1).sum(dim=1)
            if wrapper.contribution is None:
                wrapper.contribution = contribution
            else:
                wrapper.contribution += contribution

        self.iterations += 1


class TaylorFOWeightFilterGlobalPruner(GradientRankFilterPruner):
    """
    A structured pruning algorithm that prunes the filters with the smallest
    importance approximations based on the first order taylor expansion on the weight.
    Molchanov, Pavlo and Mallya, Arun and Tyree, Stephen and Frosio, Iuri and Kautz, Jan,
    "Importance Estimation for Neural Network Pruning", CVPR 2019.
    http://jankautz.com/publications/Importance4NNPruning_CVPR19.pdf
    """

    def __init__(self, model, config_list, optimizer, statistics_batch_num=1):
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
        statistics_batch_num : int
            Num of batches for activation statistics
        """
        super().__init__(model, config_list, optimizer, statistics_batch_num)
        self.model_sparsity = 0.
        self.pruned_step = 0
        self.prune_frequency = config_list[0]['prune_frequency']
        self.prune_filters_each_step = config_list[0]['prune_filters_each_step']

    def get_mask(self, base_mask, contribution):
        """
        Calculate the mask of given layer.
        Filters with the smallest importance approximations are masked.

        Parameters
        ----------
        base_mask : dict
            The basic mask with the same shape of weight, all item in the basic mask is 1.
        contribution : torch.Tensor
            Layer's importance approximations
        num_prune : int
            Num of filters to prune

        Returns
        -------
        dict
            dictionary for storing masks
        """

        mask = torch.gt(contribution, self.global_threshold).type_as(base_mask['weight_mask']) 
        base_mask['weight_mask'] = mask[:, None, None, None]
        if base_mask['bias_mask'] is not None:
            base_mask['bias_mask'] = torch.gt(contribution, self.global_threshold).type_as(base_mask['bias_mask']) 
        return base_mask
    
    def _count_filters(self):
        total_pruend_filters, total_filters = 0, 0
        for wrapper in self.get_modules_wrapper():
            filters = wrapper.module.weight.data.size(0)
            mask_sum = wrapper.weight_mask.view(filters, -1).sum(-1)
            pruned_filters = int(torch.sum(mask_sum == 0))
            total_filters += filters
            total_pruend_filters += pruned_filters

        return total_pruend_filters, total_filters

    def calc_contributions(self):
        """
        Calculate the estimated importance of filters as a sum of individual contribution
        based on the first order taylor expansion.
        """
        if self.model_sparsity >= self.config_list[0]['sparsity']:
            return

        if self.iterations % self.prune_frequency == 0:
            contribution_list = []
            for wrapper in self.get_modules_wrapper():
                filters = wrapper.module.weight.size(0)
                contribution = (wrapper.module.weight*wrapper.module.weight.grad).data.pow(2).view(filters, -1).sum(dim=1)
                if wrapper.contribution is None:
                    wrapper.contribution = contribution
                else:
                    wrapper.contribution += contribution

                mask_sum = wrapper.weight_mask.view(filters, -1).sum(-1)
                wrapper.contribution = torch.where(mask_sum == 0, torch.zeros_like(wrapper.contribution), wrapper.contribution)
                contribution_list.append(wrapper.contribution.clone() / (self.iterations+1))

            all_contributions = torch.cat(contribution_list)
            pruned_filters, self.total_filters = self._count_filters()
            self.prune_filters_now = pruned_filters + self.prune_filters_each_step
            print('pruned_filters', pruned_filters, 'self.prune_filters_each_step', self.prune_filters_each_step,  'self.prune_filters_now', self.prune_filters_now)
            self.model_sparsity = self.prune_filters_now / self.total_filters
            self.global_threshold = torch.topk(all_contributions.view(-1), self.prune_filters_now, largest=False)[0].max()
            self.pruned_step += 1
            print(self.global_threshold)
            print('pruned_step',self.pruned_step, 'prune_filters_now',self.prune_filters_now, 'total_filters',self.total_filters, 'model_sparsity',self.model_sparsity)
        
        self.iterations += 1

    def calc_mask(self, wrapper, **kwargs):
        """
        Calculate the mask of given layer.
        Filters with the smallest importance criterion which is calculated from the activation are masked.

        Parameters
        ----------
        wrapper : Module
            the layer to instrument the compression operation

        Returns
        -------
        dict
            dictionary for storing masks
        """

        weight = wrapper.module.weight.data
        op_type = wrapper.type
        config = wrapper.config
        assert 0 <= config.get('sparsity') < 1, "sparsity must in the range [0, 1)"
        assert op_type in config.get('op_types')

        if (self.iterations-1) % self.prune_frequency != 0 or wrapper.if_calculated: 
            return None

        mask = {'weight_mask': wrapper.weight_mask, 'bias_mask': wrapper.bias_mask}
        filters = weight.size(0)
        if filters >= 2:
            mask = self.get_mask(mask, wrapper.contribution / (self.iterations+1))
            if self.model_sparsity >= config.get('sparsity'):
                wrapper.if_calculated = True

        return mask