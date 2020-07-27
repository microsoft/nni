# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import numpy as np
import tensorflow as tf
from .compressor import Pruner

__all__ = ['LevelPruner', 'AGPPruner', 'FPGMPruner']

_logger = logging.getLogger(__name__)


class LevelPruner(Pruner):
    """
    Parameters
    ----------
    model : tensorflow model
        Model to be pruned
    config_list : list
        Supported keys:
            - sparsity : This is to specify the sparsity operations to be compressed to.
            - op_types : Operation types to prune.
    """
    def __init__(self, model, config_list):
        super().__init__(model, config_list)
        self.mask_list = {}
        self.if_init_list = {}

    def calc_mask(self, layer, config):
        weight = layer.weight
        op_name = layer.name
        if self.if_init_list.get(op_name, True):
            threshold = tf.contrib.distributions.percentile(tf.abs(weight), config['sparsity'] * 100)
            mask = tf.cast(tf.math.greater(tf.abs(weight), threshold), weight.dtype)
            self.mask_list.update({op_name: mask})
            self.if_init_list.update({op_name: False})
        else:
            mask = self.mask_list[op_name]
        return mask


class AGPPruner(Pruner):
    """
    Parameters
    ----------
    model : torch.nn.module
        Model to be pruned.
    config_list : listlist
        Supported keys:
            - initial_sparsity: This is to specify the sparsity when compressor starts to compress.
            - final_sparsity: This is to specify the sparsity when compressor finishes to compress.
            - start_epoch: This is to specify the epoch number when compressor starts to compress, default start from epoch 0.
            - end_epoch: This is to specify the epoch number when compressor finishes to compress.
            - frequency: This is to specify every *frequency* number epochs compressor compress once, default frequency=1.
    """

    def __init__(self, model, config_list):
        super().__init__(model, config_list)
        self.mask_list = {}
        self.if_init_list = {}
        self.now_epoch = tf.Variable(0)
        self.assign_handler = []

    def calc_mask(self, layer, config):
        weight = layer.weight
        op_name = layer.name
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
        for k in self.if_init_list:
            self.if_init_list[k] = True


class FPGMPruner(Pruner):
    """
    Parameters
    ----------
    model : tensorflow model
        Model to be pruned
    config_list : list
        Supported keys:
            - sparsity : percentage of convolutional filters to be pruned.
            - op_types : Only Conv2d is supported in FPGM Pruner.
    """
    def __init__(self, model, config_list):
        super().__init__(model, config_list)
        self.mask_dict = {}
        self.assign_handler = []
        self.epoch_pruned_layers = set()

    def calc_mask(self, layer, config):
        """
        Supports Conv1D, Conv2D
        filter dimensions for Conv1D:
        LEN: filter length
        IN: number of input channel
        OUT: number of output channel

        filter dimensions for Conv2D:
        H: filter height
        W: filter width
        IN: number of input channel
        OUT: number of output channel

        Parameters
        ----------
        layer : LayerInfo
            calculate mask for `layer`'s weight
        config : dict
            the configuration for generating the mask
        """

        weight = layer.weight
        op_type = layer.type
        op_name = layer.name
        assert 0 <= config.get('sparsity') < 1
        assert op_type in ['Conv1D', 'Conv2D']
        assert op_type in config['op_types']

        if layer.name in self.epoch_pruned_layers:
            assert layer.name in self.mask_dict
            return self.mask_dict.get(layer.name)

        try:
            w = tf.stop_gradient(tf.transpose(tf.reshape(weight, (-1, weight.shape[-1])), [1, 0]))
            masks = np.ones(w.shape)
            num_filters = w.shape[0]
            num_prune = int(num_filters * config.get('sparsity'))
            if num_filters < 2 or num_prune < 1:
                return masks
            min_gm_idx = self._get_min_gm_kernel_idx(w, num_prune)

            for idx in min_gm_idx:
                masks[idx] = 0.
        finally:
            masks = tf.reshape(tf.transpose(masks, [1, 0]), weight.shape)
            masks = tf.Variable(masks)
            self.mask_dict.update({op_name: masks})
            self.epoch_pruned_layers.add(layer.name)

        return masks

    def _get_min_gm_kernel_idx(self, weight, n):
        dist_list = []
        for out_i in range(weight.shape[0]):
            dist_sum = self._get_distance_sum(weight, out_i)
            dist_list.append((dist_sum, out_i))
        min_gm_kernels = sorted(dist_list, key=lambda x: x[0])[:n]
        return [x[1] for x in min_gm_kernels]

    def _get_distance_sum(self, weight, out_idx):
        anchor_w = tf.tile(tf.expand_dims(weight[out_idx], 0), [weight.shape[0], 1])
        x = weight - anchor_w
        x = tf.math.reduce_sum((x*x), -1)
        x = tf.math.sqrt(x)
        return tf.math.reduce_sum(x)

    def update_epoch(self, epoch):
        self.epoch_pruned_layers = set()
