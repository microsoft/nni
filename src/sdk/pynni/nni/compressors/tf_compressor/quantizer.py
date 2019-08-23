import tensorflow as tf
from ._nnimc_tf import TfQuantizer
from ._nnimc_tf import _tf_default_get_configure, _tf_default_load_configure_file

class NaiveQuantizer(TfQuantizer):
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
    
    def load_configure(self, config_path):
        config_list = _tf_default_load_configure_file(config_path, 'QATquantizer')
        for config in config_list.get('config', []):
            self.configure_list.append(config)
        
    def get_qbits(self, configure={}):
        sparsity = configure.get('q_bits',0)
        return sparsity

    def quantize_weight(self, layer_info, weight):
        q_bits = self.get_qbits(_tf_default_get_configure(self.configure_list, layer_info))

        a = tf.stop_gradient(tf.reduce_min(weight))
        b = tf.stop_gradient(tf.reduce_max(weight))
        n = tf.cast(2 ** q_bits, tf.float32)
        scale = b-a/(n-1)
        
        with tf.get_default_graph().gradient_override_map({'Round': 'Identity'}):
            qw = tf.round((weight-a)/scale)*scale +a
        
        return qw

class DoReFaQuantizer(TfQuantizer):
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

    def load_configure(self, config_path):
        config_list = _tf_default_load_configure_file(config_path, 'DoReFaQuantizer')
        for config in config_list.get('config', []):
            self.configure_list.append(config)
        
    def get_qbits(self, configure={}):
        sparsity = configure.get('q_bits',0)
        return sparsity

    def quantize_weight(self, layer_info, weight):
        q_bits = self.get_qbits(_tf_default_get_configure(self.configure_list, layer_info))
        a = tf.math.tanh(weight)
        b = a/(2*tf.reduce_max(tf.abs(weight))) + 0.5

        scale = pow(2, q_bits-1)
        with tf.get_default_graph().gradient_override_map({'Round': 'Identity'}):
            qw = tf.round(b*scale)/scale
        r_qw = 2*qw - 1
        return r_qw
