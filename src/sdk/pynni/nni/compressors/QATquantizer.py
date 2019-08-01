try:
    import torch
    from nnimc import TorchQuantizer

    class TorchDoReFaQuantizer(TorchQuantizer):
        def __init__(self, q_bits):
            super().__init__()
            self.q_bits = q_bits
        
        def quantize_weight(self, layer_info, weight):
            out = weight.tanh()
            out = out /( 2 * out.abs().max()) + 0.5
            out = self.quantize(out, self.q_bits)
            out = 2 * out -1
            #print(weight,out)
            return out
        
        def quantize(self, input_ri, q_bits):
            scale = pow(2, q_bits)-1
            output = torch.round(input_ri*scale)/scale
            #input_ri.mul_(scale).round_()
            #input_ri.div_(scale)
            return output

    class TorchQATquantizer(TorchQuantizer):
        def __init__(self, q_bits):
            super().__init__()
            self.q_bits = q_bits
        
        def quantize_weight(self, layer_info, weight):
            if self.q_bits <= 1:
                return weight
            a = torch.min(weight)
            b = torch.max(weight)
            n = pow(2,self.q_bits)
            scale = (b-a)/(n-1)
            zero_point = a
            #print(a,b,scale)
            out = torch.round((weight - zero_point)/scale)
            out = out*scale + zero_point
            orig_type = weight.dtype
            #print(weight,out)
            return out.type(orig_type)

except ModuleNotFoundError:
    pass



try:
    import tensorflow as tf
    from nnimc import TfQuantizer
    from tensorflow.python.framework import ops as tf_ops

    class TfQATquantizer(TfQuantizer):
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

    class TfDoReFaQuantizer(TfQuantizer):
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
        


except ModuleNotFoundError:
    pass