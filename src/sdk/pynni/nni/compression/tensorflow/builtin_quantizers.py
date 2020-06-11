# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import tensorflow as tf
from .compressor import Quantizer

__all__ = ['NaiveQuantizer', 'QAT_Quantizer', 'DoReFaQuantizer']

_logger = logging.getLogger(__name__)


class NaiveQuantizer(Quantizer):
    """quantize weight to 8 bits
    """
    def __init__(self, model, config_list):
        super().__init__(model, config_list)
        self.layer_scale = {}

    def quantize_weight(self, weight, config, op_name, **kwargs):
        new_scale = tf.reduce_max(tf.abs(weight)) / 127
        scale = tf.maximum(self.layer_scale.get(op_name, tf.constant(0.0)), new_scale)
        self.layer_scale[op_name] = scale
        orig_type = weight.dtype
        return tf.cast(tf.cast(weight / scale, tf.int8), orig_type) * scale


class QAT_Quantizer(Quantizer):
    """Quantizer using the DoReFa scheme, as defined in:
    Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference
    http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf
    """
    def __init__(self, model, config_list):
        """
        config_list: supported keys:
            - q_bits
        """
        super().__init__(model, config_list)

    def quantize_weight(self, weight, config, **kwargs):
        a = tf.stop_gradient(tf.reduce_min(weight))
        b = tf.stop_gradient(tf.reduce_max(weight))
        n = tf.cast(2 ** config['q_bits'], tf.float32)
        scale = b-a/(n-1)

        # use gradient_override_map to change round to idetity for gradient
        with tf.get_default_graph().gradient_override_map({'Round': 'Identity'}):
            qw = tf.round((weight-a)/scale)*scale +a

        return qw


class DoReFaQuantizer(Quantizer):
    """Quantizer using the DoReFa scheme, as defined in:
    Zhou et al., DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients
    (https://arxiv.org/abs/1606.06160)
    """
    def __init__(self, model, config_list):
        """
        config_list: supported keys:
            - q_bits
        """
        super().__init__(model, config_list)

    def quantize_weight(self, weight, config, **kwargs):
        a = tf.math.tanh(weight)
        b = a/(2*tf.reduce_max(tf.abs(weight))) + 0.5

        scale = pow(2, config['q_bits'] - 1)
        # use gradient_override_map to change round to idetity for gradient
        with tf.get_default_graph().gradient_override_map({'Round': 'Identity'}):
            qw = tf.round(b*scale)/scale
        r_qw = 2 * qw - 1
        return r_qw
