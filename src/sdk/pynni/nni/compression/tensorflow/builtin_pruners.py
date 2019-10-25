import logging
import tensorflow as tf
from .compressor import Pruner

__all__ = ['LevelPruner', 'AGP_Pruner', 'FPGMPruner']

_logger = logging.getLogger(__name__)


class LevelPruner(Pruner):
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
            threshold = tf.contrib.distributions.percentile(tf.abs(weight), config['sparsity'] * 100)
            mask = tf.cast(tf.math.greater(tf.abs(weight), threshold), weight.dtype)
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
            - start_epoch: start epoch numer begin update mask
            - end_epoch: end epoch number stop update mask
            - frequency: if you want update every 2 epoch, you can set it 2
        """
        super().__init__(config_list)
        self.mask_list = {}
        self.if_init_list = {}
        self.now_epoch = tf.Variable(0)
        self.assign_handler = []

    def calc_mask(self, weight, config, op_name, **kwargs):
        print('config:', config)
        print('kwargs:', kwargs)
        print('op_name:', op_name)
        start_epoch = config.get('start_epoch', 0)
        freq = config.get('frequency', 1)
        if self.now_epoch >= start_epoch and self.if_init_list.get(op_name, True) and (
                self.now_epoch - start_epoch) % freq == 0:
            target_sparsity = self.compute_target_sparsity(config)
            threshold = tf.contrib.distributions.percentile(weight, target_sparsity * 100)
            # stop gradient in case gradient change the mask
            mask = tf.stop_gradient(tf.cast(tf.math.greater(weight, threshold), weight.dtype))
            self.assign_handler.append(tf.assign(weight, weight * mask))
            self.mask_list.update({op_name: tf.constant(mask)})
            self.if_init_list.update({op_name: False})
        else:
            mask = self.mask_list[op_name]
        return mask

    def compute_target_sparsity(self, config):
        end_epoch = config.get('end_epoch', 1)
        start_epoch = config.get('start_epoch', 0)
        freq = config.get('frequency', 1)
        final_sparsity = config.get('final_sparsity', 0)
        initial_sparsity = config.get('initial_sparsity', 0)

        if end_epoch <= start_epoch or initial_sparsity >= final_sparsity:
            _logger.warning('your end epoch <= start epoch or initial_sparsity >= final_sparsity')
            return final_sparsity

        now_epoch = tf.minimum(self.now_epoch, tf.constant(end_epoch))
        span = int(((end_epoch - start_epoch - 1) // freq) * freq)
        assert span > 0
        base = tf.cast(now_epoch - start_epoch, tf.float32) / span
        target_sparsity = (final_sparsity +
                           (initial_sparsity - final_sparsity) *
                           (tf.pow(1.0 - base, 3)))
        return target_sparsity

    def update_epoch(self, epoch, sess):
        sess.run(self.assign_handler)
        sess.run(tf.assign(self.now_epoch, int(epoch)))
        for k in self.if_init_list.keys():
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
            - start_epoch: start epoch number begin update mask
            - end_epoch: end epoch number stop update mask, you should make sure start_epoch <= end_epoch
        """
        super().__init__(config_list)
        self.mask_list = {}

    def calc_mask(self, conv_kernel_weight, config, op, op_type, op_name):
        print('config:', config)
        print('op:', op)
        print('op_type:', op_type)
        print('op_name:', op_name)
        assert 0 <= config.get('pruning_rate') < 1
        assert config['op_type'] in ['Conv1D', 'Conv2D', 'Conv3D']

        weight = tf.stop_gradient(conv_kernel_weight)
        masks = tf.ones_like(weight)

        if op_type == config['op_type']:
            num_kernels = weight.shape[0].value * weight.shape[1].value
            num_prune = int(num_kernels * config.get('pruning_rate'))
            if num_kernels < 2 or num_prune < 1:
                self.mask_list.update({op_name: masks})
                return masks
            min_gm_idx = self._get_min_gm_kernel_idx(weight, num_prune)
            for idx in min_gm_idx:
                masks[idx] = 0.
 
        self.mask_list.update({op_name: masks})
        return masks

    def _get_min_gm_kernel_idx(self, weight, n):
        """supports Conv1D, Conv2D, Conv3D
        filter/kernel dimensions for Conv2d:
        IN: number of input channel
        OUT: number of output channel
        H: filter height
        W: filter width
        """
        assert len(weight.shape) >= 3
        assert weight.shape[0].value * weight.shape[1].value > 2

        dist_list = []
        for in_i in range(weight.shape[0].value):
            for out_i in range(weight.shape[1].value):
                dist_sum = self._get_distance_sum_fast(weight, in_i, out_i)
                dist_list.append((dist_sum, (in_i, out_i)))
        min_gm_kernels = sorted(dist_list, key=lambda x: x[0])[:n]
        return [x[1] for x in min_gm_kernels]

    def _get_distance_sum_fast(self, weight, in_idx, out_idx):
        w = tf.reshape(weight, (-1, weight.shape[-2].value, weight.shape[-1].value))
        anchor_w = tf.tile(tf.expand_dims(weight[in_idx, out_idx], 0), [w.shape[0].value, 1, 1])
        x = w - anchor_w
        x = tf.math.reduce_sum((x*x), (-2, -1))
        x = tf.math.sqrt(x)
        return tf.math.reduce_sum(x)
