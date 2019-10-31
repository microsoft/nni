import logging
import torch
from .compressor import Pruner

__all__ = ['LevelPruner', 'AGP_Pruner', 'FPGMPruner']

logger = logging.getLogger('torch pruner')


class LevelPruner(Pruner):
    """Prune to an exact pruning level specification
    """

    def __init__(self, config_list):
        """
        config_list: supported keys:
            - sparsity
        """
        super().__init__(config_list)
        self.mask_list = {}
        self.if_init_list = {}

    def calc_mask(self, weight, config, op_name, **kwargs):
        if self.if_init_list.get(op_name, True):
            w_abs = weight.abs()
            k = int(weight.numel() * config['sparsity'])
            if k == 0:
                return torch.ones(weight.shape).type_as(weight)
            threshold = torch.topk(w_abs.view(-1), k, largest=False).values.max()
            mask = torch.gt(w_abs, threshold).type_as(weight)
            self.mask_list.update({op_name: mask})
            self.if_init_list.update({op_name: False})
        else:
            mask = self.mask_list[op_name]
        return mask


class AGP_Pruner(Pruner):
    """An automated gradual pruning algorithm that prunes the smallest magnitude
    weights to achieve a preset level of network sparsity.

    Michael Zhu and Suyog Gupta, "To prune, or not to prune: exploring the
    efficacy of pruning for model compression", 2017 NIPS Workshop on Machine
    Learning of Phones and other Consumer Devices,
    https://arxiv.org/pdf/1710.01878.pdf
    """

    def __init__(self, config_list):
        """
        config_list: supported keys:
            - initial_sparsity
            - final_sparsity: you should make sure initial_sparsity <= final_sparsity
            - start_epoch: start epoch number begin update mask
            - end_epoch: end epoch number stop update mask, you should make sure start_epoch <= end_epoch
            - frequency: if you want update every 2 epoch, you can set it 2
        """
        super().__init__(config_list)
        self.mask_list = {}
        self.now_epoch = 0
        self.if_init_list = {}

    def calc_mask(self, weight, config, op_name, **kwargs):
        start_epoch = config.get('start_epoch', 0)
        freq = config.get('frequency', 1)
        if self.now_epoch >= start_epoch and self.if_init_list.get(op_name, True) and (
                self.now_epoch - start_epoch) % freq == 0:
            mask = self.mask_list.get(op_name, torch.ones(weight.shape).type_as(weight))
            target_sparsity = self.compute_target_sparsity(config)
            k = int(weight.numel() * target_sparsity)
            if k == 0 or target_sparsity >= 1 or target_sparsity <= 0:
                return mask
            # if we want to generate new mask, we should update weigth first
            w_abs = weight.abs() * mask
            threshold = torch.topk(w_abs.view(-1), k, largest=False).values.max()
            new_mask = torch.gt(w_abs, threshold).type_as(weight)
            self.mask_list.update({op_name: new_mask})
            self.if_init_list.update({op_name: False})
        else:
            new_mask = self.mask_list.get(op_name, torch.ones(weight.shape).type_as(weight))
        return new_mask

    def compute_target_sparsity(self, config):
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
        if epoch > 0:
            self.now_epoch = epoch
            for k in self.if_init_list:
                self.if_init_list[k] = True

class FPGMPruner(Pruner):
    """A filter pruner via geometric median.
    "Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration", 
    https://arxiv.org/pdf/1811.00250.pdf
    """

    def __init__(self, config_list):
        """
        config_list: supported keys:
            - pruning_rate: percentage of convolutional filters to be pruned.
        """
        super().__init__(config_list)
        self.mask_list = {}

    def calc_mask(self, weight, config, op, op_type, op_name, **kwargs):
        """supports Conv1d, Conv2d, Conv3d
        filter dimensions for Conv1d:
        IN: number of input channel
        OUT: number of output channel
        LEN: filter length

        filter dimensions for Conv2d:
        IN: number of input channel
        OUT: number of output channel
        H: filter height
        W: filter width
        """

        assert 0 <= config.get('pruning_rate') < 1
        assert op_type in ['Conv1d', 'Conv2d', 'Conv3d']
        assert op_type in config['op_types']

        if op_name in self.epoch_pruned_layers:
            assert op_name in self.mask_list
            return self.mask_list.get(op_name)

        masks = torch.ones(weight.size())

        try:
            num_kernels = weight.size(0) * weight.size(1)
            num_prune = int(num_kernels * config.get('pruning_rate'))
            if num_kernels < 2 or num_prune < 1:
                return masks
            min_gm_idx = self._get_min_gm_kernel_idx(weight, num_prune)
            for idx in min_gm_idx:
                masks[idx] = 0.
        finally:
            self.mask_list.update({op_name: masks})
            self.epoch_pruned_layers.add(op_name)

        return masks

    def _get_min_gm_kernel_idx(self, weight, n):
        assert len(weight.size()) >= 3
        assert weight.size(0) * weight.size(1) > 2

        dist_list = []
        for in_i in range(weight.size(0)):
            for out_i in range(weight.size(1)):
                dist_sum = self._get_distance_sum(weight, in_i, out_i)
                dist_list.append((dist_sum, (in_i, out_i)))
        min_gm_kernels = sorted(dist_list, key=lambda x: x[0])[:n]
        return [x[1] for x in min_gm_kernels]

    def _get_distance_sum(self, weight, in_idx, out_idx):
        """ Optimized verision of following naive implementation:
        def _get_distance_sum(self, weight, in_idx, out_idx):
            w = weight.view(-1, weight.size(-2), weight.size(-1))
            dist_sum = 0.
            for k in w:
                dist_sum += torch.dist(k, weight[in_idx, out_idx], p=2)
            return dist_sum
        """
        w = weight.view(-1, weight.size(-2), weight.size(-1))
        anchor_w = weight[in_idx, out_idx].unsqueeze(0).expand(w.size(0), w.size(1), w.size(2))
        x = w - anchor_w
        x = (x*x).sum((-2,-1))
        x = torch.sqrt(x)
        return x.sum()

    def update_epoch(self, epoch):
        self.epoch_pruned_layers = set()
