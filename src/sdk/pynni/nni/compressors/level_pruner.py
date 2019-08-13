try:
    import torch
    from nnimc import TorchPruner
    
    class TorchLevelPruner(TorchPruner):
        def __init__(self, sparsity = 0, layer_sparsity = { }):
            super().__init__()
            self.default_sparsity = sparsity
            self.layer_sparsity = layer_sparsity
    
        def calc_mask(self, layer_info, weight):
            sparsity = self.layer_sparsity.get(layer_info.name, self.default_sparsity)
            w_abs = weight.abs()
            k = int(weight.numel() * sparsity)
            threshold = torch.topk(w_abs.view(-1), k, largest = False).values.max()
            return torch.gt(w_abs, threshold).type(weight.type())

except ImportError:
    pass


try:
    import tensorflow as tf
    from nnimc import TfPruner

    class TfLevelPruner(TfPruner):
        def __init__(self, sparsity = 0, layer_sparsity = { }):
            super().__init__()
            self.default_sparsity = sparsity
            self.layer_sparsity = layer_sparsity

        def calc_mask(self, layer_info, weight):
            sparsity = self.layer_sparsity.get(layer_info.name, self.default_sparsity)
            threshold = tf.contrib.distributions.percentile(weight.abs(), sparsity * 100)
            return tf.cast(tf.math.greater(weight.abs(), threshold), weight.dtype)

except ImportError:
    pass
