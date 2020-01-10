# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import logging
import torch
from .compressor import Pruner

__all__ = ['LevelPruner', 'AGP_Pruner', 'SlimPruner', 'LotteryTicketPruner']

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
            assert op_name in self.mask_dict, "op_name not in the mask_dict"
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

class LotteryTicketPruner(Pruner):
    """
    This is a Pytorch implementation of the paper "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks",
    following NNI model compression interface.

    1. Randomly initialize a neural network f(x;theta_0) (where theta_0 follows D_{theta}).
    2. Train the network for j iterations, arriving at parameters theta_j.
    3. Prune p% of the parameters in theta_j, creating a mask m.
    4. Reset the remaining parameters to their values in theta_0, creating the winning ticket f(x;m*theta_0).
    5. Repeat step 2, 3, and 4.
    """

    def __init__(self, model, config_list, optimizer, lr_scheduler=None, reset_weights=True):
        """
        Parameters
        ----------
        model : pytorch model
            The model to be pruned
        config_list : list
            Supported keys:
                - prune_iterations : The number of rounds for the iterative pruning.
                - sparsity : The final sparsity when the compression is done.
        optimizer : pytorch optimizer
            The optimizer for the model
        lr_scheduler : pytorch lr scheduler
            The lr scheduler for the model if used
        reset_weights : bool
            Whether reset weights and optimizer at the beginning of each round.
        """
        super().__init__(model, config_list)
        self.curr_prune_iteration = None
        self.prune_iterations = self._validate_config(config_list)

        # save init weights and optimizer
        self.reset_weights = reset_weights
        if self.reset_weights:
            self._model = model
            self._optimizer = optimizer
            self._model_state = copy.deepcopy(model.state_dict())
            self._optimizer_state = copy.deepcopy(optimizer.state_dict())
            self._lr_scheduler = lr_scheduler
            if lr_scheduler is not None:
                self._scheduler_state = copy.deepcopy(lr_scheduler.state_dict())

    def _validate_config(self, config_list):
        prune_iterations = None
        for config in config_list:
            assert 'prune_iterations' in config, 'prune_iterations must exist in your config'
            assert 'sparsity' in config, 'sparsity must exist in your config'
            if prune_iterations is not None:
                assert prune_iterations == config[
                    'prune_iterations'], 'The values of prune_iterations must be equal in your config'
            prune_iterations = config['prune_iterations']
        return prune_iterations

    def _print_masks(self, print_mask=False):
        torch.set_printoptions(threshold=1000)
        for op_name in self.mask_dict.keys():
            mask = self.mask_dict[op_name]
            print('op name: ', op_name)
            if print_mask:
                print('mask: ', mask)
            # calculate current sparsity
            mask_num = mask['weight'].sum().item()
            mask_size = mask['weight'].numel()
            print('sparsity: ', 1 - mask_num / mask_size)
        torch.set_printoptions(profile='default')

    def _calc_sparsity(self, sparsity):
        keep_ratio_once = (1 - sparsity) ** (1 / self.prune_iterations)
        curr_keep_ratio = keep_ratio_once ** self.curr_prune_iteration
        return max(1 - curr_keep_ratio, 0)

    def _calc_mask(self, weight, sparsity, op_name):
        if self.curr_prune_iteration == 0:
            mask = torch.ones(weight.shape).type_as(weight)
        else:
            curr_sparsity = self._calc_sparsity(sparsity)
            assert self.mask_dict.get(op_name) is not None
            curr_mask = self.mask_dict.get(op_name)
            w_abs = weight.abs() * curr_mask['weight']
            k = int(w_abs.numel() * curr_sparsity)
            threshold = torch.topk(w_abs.view(-1), k, largest=False).values.max()
            mask = torch.gt(w_abs, threshold).type_as(weight)
        return {'weight': mask}

    def calc_mask(self, layer, config):
        """
        Generate mask for the given ``weight``.

        Parameters
        ----------
        layer : LayerInfo
            The layer to be pruned
        config : dict
            Pruning configurations for this weight

        Returns
        -------
        tensor
            The mask for this weight
        """
        assert self.mask_dict.get(layer.name) is not None, 'Please call iteration_start before training'
        mask = self.mask_dict[layer.name]
        return mask

    def get_prune_iterations(self):
        """
        Return the range for iterations.
        In the first prune iteration, masks are all one, thus, add one more iteration

        Returns
        -------
        list
            A list for pruning iterations
        """
        return range(self.prune_iterations + 1)

    def prune_iteration_start(self):
        """
        Control the pruning procedure on updated epoch number.
        Should be called at the beginning of the epoch.
        """
        if self.curr_prune_iteration is None:
            self.curr_prune_iteration = 0
        else:
            self.curr_prune_iteration += 1
        assert self.curr_prune_iteration < self.prune_iterations + 1, 'Exceed the configured prune_iterations'

        modules_to_compress = self.detect_modules_to_compress()
        for layer, config in modules_to_compress:
            sparsity = config.get('sparsity')
            mask = self._calc_mask(layer.module.weight.data, sparsity, layer.name)
            self.mask_dict.update({layer.name: mask})
        self._print_masks()

        # reinit weights back to original after new masks are generated
        if self.reset_weights:
            self._model.load_state_dict(self._model_state)
            self._optimizer.load_state_dict(self._optimizer_state)
            if self._lr_scheduler is not None:
                self._lr_scheduler.load_state_dict(self._scheduler_state)
