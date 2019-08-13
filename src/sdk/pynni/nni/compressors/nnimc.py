try:
    import torch
    from nni.compressors.torchCompressor._nnimc_torch import *
    _torch_available = True
except ModuleNotFoundError:
    _torch_available = False

try:
    import tensorflow as tf
    from nni.compressors.tfCompressor._nnimc_tf import *
    _tf_available = True
except ModuleNotFoundError:
    _tf_available = False


def detect_prunable_layers(model):
    """
    Search for all prunable layers in the model.
    This can be useful to create search space.
    """
    if _torch_available and isinstance(model, torch.nn.Model):
        return _torch_detect_prunable_layers(model)
    if _tf_available and isinstance(model, tf.Graph):
        return _tf_detect_prunable_layers(model)
