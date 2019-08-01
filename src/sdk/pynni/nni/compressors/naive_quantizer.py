try:
    import torch
    from nnimc import TorchQuantizer
    
    class TorchNaiveQuantizer(TorchQuantizer):
        def __init__(self):
            super().__init__()
            self.layer_scale = { }
    
        def quantize_weight(self, layer_info, weight):
            new_scale = weight.abs().max() / 127
            # TODO: use int id
            scale = max(self.layer_scale.get(layer_info.name, 0), new_scale)
            self.layer_scale[layer_info.name] = scale
            orig_type = weight.type()  # TODO: user layer_info
            return weight.div(scale).type(torch.int8).type(orig_type).mul(scale)

except ModuleNotFoundError:
    pass


try:
    import tensorflow as tf
    from nnimc import TfQuantizer

    class TfNaiveQuantizer(TfQuantizer):
        def __init__(self):
            super().__init__()
            self.layer_scale = { }

        def quantize_weight(self, layer_info, weight):
            new_scale = tf.reduce_max(tf.abs(weight)) / 127
            scale = tf.maximum(self.layer_scale.get(layer_info.name, tf.constant(0.0)), new_scale)
            self.layer_scale[layer_info.name] = scale
            orig_type = weight.dtype
            return tf.cast(tf.cast(weight / scale, tf.int8), orig_type) * scale

except ModuleNotFoundError:
    pass
