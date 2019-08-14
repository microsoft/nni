try:
    import tensorflow as tf
    from ._nnimc_tf import TfQuantizer

    class NaiveQuantizer(TfQuantizer):
        def __init__(self):
            super().__init__()
            self.layer_scale = { }

        def quantize_weight(self, layer_info, weight):
            new_scale = tf.reduce_max(tf.abs(weight)) / 127
            scale = tf.maximum(self.layer_scale.get(layer_info.name, tf.constant(0.0)), new_scale)
            self.layer_scale[layer_info.name] = scale
            orig_type = weight.dtype
            return tf.cast(tf.cast(weight / scale, tf.int8), orig_type) * scale
    
    class QATquantizer(TfQuantizer):
        def __init__(self, q_bits):
            super().__init__()
            self.q_bits = q_bits

        def quantize_weight(self, layer_info, weight):
            a = tf.stop_gradient(tf.reduce_min(weight))
            b = tf.stop_gradient(tf.reduce_max(weight))
            n = tf.cast(2 ** self.q_bits, tf.float32)
            scale = b-a/(n-1)
            
            with tf.get_default_graph().gradient_override_map({'Round': 'Identity'}):
                qw = tf.round((weight-a)/scale)*scale +a
            
            return qw
    
    class DoReFaQuantizer(TfQuantizer):
        def __init__(self, q_bits):
            super().__init__()
            self.q_bits = q_bits

        def quantize_weight(self, layer_info, weight):
            a = tf.math.tanh(weight)
            b = a/(2*tf.reduce_max(tf.abs(weight))) + 0.5

            scale = pow(2, self.q_bits-1)
            with tf.get_default_graph().gradient_override_map({'Round': 'Identity'}):
                qw = tf.round(b*scale)/scale
            r_qw = 2*qw - 1
            return r_qw

except ImportError:
    pass
