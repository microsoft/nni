import logging
import tensorflow as tf
from .compressor import Pruner
from tensorflow.python.ops import math_ops

__all__ = ['LevelPruner', 'AGP_Pruner']

_logger = logging.getLogger(__name__)


class LevelPruner(Pruner):
    def __init__(self, config_list):
        """
        config_list: supported keys:
            - sparsity
        """
        super().__init__(config_list)
        self.assign_handler = []
        self.now_epoch = tf.Variable(0)

    def calc_mask(self, weight, config, op_name, **kwargs):
        new_mask = tf.Variable(tf.cast(tf.ones(tf.shape(weight)), tf.float32))
        threshold = tf.contrib.distributions.percentile(tf.abs(weight), config['sparsity'] * 100)
        mask = tf.stop_gradient(tf.cast(tf.math.greater(tf.abs(weight), threshold), weight.dtype))

        def maybe_update():
            return math_ops.equal(self.now_epoch, tf.constant(0))

        def update_mask():
            return tf.assign(new_mask, mask)

        def no_update_mask():
            return tf.identity(new_mask)

        new_mask = tf.stop_gradient(tf.cond(maybe_update(), update_mask, no_update_mask))
        self.assign_handler.append(tf.assign(weight, weight * new_mask))
        return new_mask

    def update_epoch(self, epoch, sess):
        sess.run(tf.assign(self.now_epoch, int(epoch)))
        sess.run(self.assign_handler)


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
        self.now_epoch = tf.Variable(0)
        self.assign_handler = []

    def calc_mask(self, weight, config, op_name, **kwargs):
        start_epoch = config.get('start_epoch', 0)
        freq = config.get('frequency', 1)
        new_mask = tf.Variable(tf.cast(tf.ones(tf.shape(weight)), tf.float32))

        target_sparsity = self.compute_target_sparsity(config)
        threshold = tf.contrib.distributions.percentile(weight, target_sparsity * 100)
        # stop gradient in case gradient change the mask
        mask = tf.stop_gradient(tf.cast(tf.math.greater(weight, threshold), weight.dtype))

        def maybe_update():
            return math_ops.logical_and(math_ops.greater_equal(self.now_epoch, start_epoch),
                                        math_ops.equal(tf.mod((self.now_epoch - start_epoch), freq), 0))

        def update_mask():
            return tf.assign(new_mask, mask)

        def no_update_mask():
            return tf.identity(new_mask)

        new_mask = tf.stop_gradient(tf.cond(maybe_update(), update_mask, no_update_mask))
        self.assign_handler.append(tf.assign(weight, weight * new_mask))

        return new_mask

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
        sess.run(tf.assign(self.now_epoch, int(epoch)))
        sess.run(self.assign_handler)
