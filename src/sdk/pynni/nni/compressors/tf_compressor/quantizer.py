import tensorflow as tf
from ._nnimc_tf import TfQuantizer
from ._nnimc_tf import _tf_default_get_configure, _tf_default_load_configure_file

import logging
logger = logging.getLogger('tensorflow quantizer')

class NaiveQuantizer(TfQuantizer):
    """
    quantize weight to 8 bits
    """
    def __init__(self):
        super().__init__()
        self.layer_scale = {}

    def quantize_weight(self, layer_info, weight):
        new_scale = tf.reduce_max(tf.abs(weight)) / 127
        scale = tf.maximum(self.layer_scale.get(layer_info.name, tf.constant(0.0)), new_scale)
        self.layer_scale[layer_info.name] = scale
        orig_type = weight.dtype
        return tf.cast(tf.cast(weight / scale, tf.int8), orig_type) * scale

class QATquantizer(TfQuantizer):
    """
    Quantizer using the DoReFa scheme, as defined in:
    Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference
    http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf
    """
    def __init__(self, configure_list):
        """
            Configure Args:
                q_bits
        """
        super().__init__()
        self.configure_list = []
        if isinstance(configure_list, list):
            for configure in configure_list:
                self.configure_list.append(configure)
        else:
            raise ValueError('please init with configure list')
    
        
    def get_qbits(self, configure):
        if not isinstance(configure, dict):
            logger.warning('WARNING: you should input a dict to get_qbits, set DEFAULT { }')
            configure = {}
        qbits = configure.get('q_bits', 32)
        if qbits == 0:
            logger.warning('WARNING: you can not set q_bits ZERO!')
            qbits = 32
        return qbits

    def quantize_weight(self, layer_info, weight):
        q_bits = self.get_qbits(_tf_default_get_configure(self.configure_list, layer_info))

        a = tf.stop_gradient(tf.reduce_min(weight))
        b = tf.stop_gradient(tf.reduce_max(weight))
        n = tf.cast(2 ** q_bits, tf.float32)
        scale = b-a/(n-1)
        
        # use gradient_override_map to change round to idetity for gradient
        with tf.get_default_graph().gradient_override_map({'Round': 'Identity'}):
            qw = tf.round((weight-a)/scale)*scale +a
        
        return qw

class DoReFaQuantizer(TfQuantizer):
    """
    Quantizer using the DoReFa scheme, as defined in:
    Zhou et al., DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients
    (https://arxiv.org/abs/1606.06160)
    """
    def __init__(self, configure_list):
        """
            Configure Args:
                q_bits
        """
        super().__init__()
        self.configure_list = []
        if isinstance(configure_list, list):
            for configure in configure_list:
                self.configure_list.append(configure)
        else:
            raise ValueError('please init with configure list')

        
    def get_qbits(self, configure):
        if not isinstance(configure, dict):
            logger.warning('WARNING: you should input a dict to get_qbits, set DEFAULT { }')
            configure = {}
        qbits = configure.get('q_bits', 32)
        if qbits == 0:
            logger.warning('WARNING: you can not set q_bits ZERO!')
            qbits = 32
        return qbits

    def quantize_weight(self, layer_info, weight):
        q_bits = self.get_qbits(_tf_default_get_configure(self.configure_list, layer_info))
        a = tf.math.tanh(weight)
        b = a/(2*tf.reduce_max(tf.abs(weight))) + 0.5

        scale = pow(2, q_bits-1)
        # use gradient_override_map to change round to idetity for gradient
        with tf.get_default_graph().gradient_override_map({'Round': 'Identity'}):
            qw = tf.round(b*scale)/scale
        r_qw = 2*qw - 1
        return r_qw
