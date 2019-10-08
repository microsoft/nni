import logging
import tensorflow as tf
from .compressor import Pruner

__all__ = [ 'LevelPruner', 'AGP_Pruner', 'SensitivityPruner' ]

_logger = logging.getLogger(__name__)


class LevelPruner(Pruner):
    def __init__(self, config_list):
        """
            Configure Args:
                sparsity
        """
        super().__init__(config_list)

    def calc_mask(self, weight, config, **kwargs):
        threshold = tf.contrib.distributions.percentile(tf.abs(weight), config['sparsity'] * 100)
        return tf.cast(tf.math.greater(tf.abs(weight), threshold), weight.dtype)


class AGP_Pruner(Pruner):
    """
    An automated gradual pruning algorithm that prunes the smallest magnitude 
    weights to achieve a preset level of network sparsity.

    Michael Zhu and Suyog Gupta, "To prune, or not to prune: exploring the
    efficacy of pruning for model compression", 2017 NIPS Workshop on Machine
    Learning of Phones and other Consumer Devices,
    https://arxiv.org/pdf/1710.01878.pdf
    """
    def __init__(self, config_list):
        """
            Configure Args:
                initial_sparsity
                final_sparsity: you should make sure initial_sparsity <= final_sparsity
                start_epoch: start epoch numer begin update mask
                end_epoch: end epoch number stop update mask
                frequency: if you want update every 2 epoch, you can set it 2
        """
        super().__init__(config_list)
        self.now_epoch = tf.Variable(0)
        self.assign_handler = []

    def calc_mask(self, weight, config, **kwargs):
        target_sparsity = self.compute_target_sparsity(config)
        threshold = tf.contrib.distributions.percentile(weight, target_sparsity * 100)
        # stop gradient in case gradient change the mask
        mask = tf.stop_gradient(tf.cast(tf.math.greater(weight, threshold), weight.dtype))
        self.assign_handler.append(tf.assign(weight, weight * mask))
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
        span = int(((end_epoch - start_epoch-1)//freq)*freq)
        assert span > 0
        base = tf.cast(now_epoch - start_epoch, tf.float32) / span
        target_sparsity = (final_sparsity + 
                            (initial_sparsity - final_sparsity)*
                            (tf.pow(1.0 - base, 3)))
        return target_sparsity

    def update_epoch(self, epoch, sess):
        sess.run(self.assign_handler)
        sess.run(tf.assign(self.now_epoch, int(epoch)))
    

class SensitivityPruner(Pruner):
    """
    Use algorithm from "Learning both Weights and Connections for Efficient Neural Networks" 
    https://arxiv.org/pdf/1506.02626v3.pdf

    I.e.: "The pruning threshold is chosen as a quality parameter multiplied
    by the standard deviation of a layers weights."
    """
    def __init__(self, config_list):
        """
            Configure Args:
                sparsity: chosen pruning sparsity
        """
        super().__init__(config_list)
        self.layer_mask = {}
        self.assign_handler = []

    def calc_mask(self, weight, config, op_name, **kwargs):
        target_sparsity = config['sparsity'] * tf.math.reduce_std(weight) 
        mask = tf.get_variable(op_name + '_mask', initializer=tf.ones(weight.shape), trainable=False)
        self.layer_mask[op_name] = mask

        weight_assign_handler = tf.assign(weight, mask*weight)
        # use control_dependencies so that weight_assign_handler will be executed before mask_update_handler
        with tf.control_dependencies([weight_assign_handler]):
            threshold = tf.contrib.distributions.percentile(weight, target_sparsity * 100)
            # stop gradient in case gradient change the mask
            new_mask = tf.stop_gradient(tf.cast(tf.math.greater(weight, threshold), weight.dtype))
            mask_update_handler = tf.assign(mask, new_mask)
            self.assign_handler.append(mask_update_handler)
        return mask

    def update_epoch(self, epoch, sess):
        sess.run(self.assign_handler)
