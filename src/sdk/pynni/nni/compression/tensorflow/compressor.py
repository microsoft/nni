# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Abstract base classes for TensorFlow model compression.
"""

import logging

import tensorflow as tf
assert tf.__version__.startswith('2'), 'NNI model compression only supports TensorFlow v2.x'

from . import default_layers

_logger = logging.getLogger(__name__)


class LayerInfo:
    """
    This structure contains all infomation needed to compress a TensorFlow ``Layer``.


    Attributes
    ----------
    layer : tf.keras.layers.Layer
        The layer.
    name : str
        The layer's name. Note that it's local to sub-model and may differ from its attribute name.
    type : str
        Name of the layer's class.
    path : list of str or tuple of (str, int)
        The layer object's and its parents' attribute name / list index.
        For example, if the path is `[('cells', 2), 'conv']`, then the layer can be accessed as `model.cells[2].conv`.
    config : JSON object
        Selected configuration for this layer. The format is detailed in tutorial.

    Parameters
    ----------
    layer : tf.keras.layers.Layer
        See attributes section.
    path : list of str or tuple of (str, int)
        See attributes section.
    """

    def __init__(self, layer, path=None):
        self.layer = layer
        self.name = layer.name
        self.type = type(layer).__name__
        self.path = path
        self.config = None


class Compressor:
    """
    Common base class for all compressors.

    This class is designed for other base classes.
    Algorithms should inherit ``Pruner`` or ``Quantizer`` instead.


    Attributes
    ----------
    bound_model : tf.keras.Model
        Compressed user model.
    wrappers : list of tf.keras.Model
        A wrapper is an instrumented TF ``Layer``, in ``Model`` format.
        The list is ordered by preorder traversal.

    Parameters
    ----------
    LayerWrapperClass : a class derive from Model
        The class used to instrument layers.
    model : tf.keras.Model
        The user model to be compressed.
    config_list : list of JSON object
        User configuration. The format is detailed in tutorial.
    """

    def __init__(self, LayerWrapperClass, model, config_list):
        assert isinstance(model, tf.keras.Model)
        if isinstance(model, tf.keras.Sequential):
            raise ValueError('NNI model compression does not support `Sequential` model for now')
        self.validate_config(model, config_list)

        self.bound_model = model
        self.wrappers = []

        for layer_info in _detect_layers_to_compress(model, config_list):
            self.wrappers.append(LayerWrapperClass(layer_info, self))
        if not self.wrappers:
            _logger.warning('Nothing is configured to compress, please check your model and config list')

        _instrument_model(model, self.wrappers)

    def set_wrappers_attribute(self, name, value):
        """
        Call ``setattr`` on all wrappers.
        """
        for wrapper in self.wrappers:
            setattr(wrapper, name, value)


class Pruner(Compressor):
    """
    Base class for pruning algorithms.

    End users should use ``compress`` and callback APIs (WIP) to prune their models.

    The underlying model is instrumented upon initialization of pruner object.
    So if you want to pre-train the model, train it before creating pruner object.

    The compressed model can only execute in eager mode.

    Algorithm developers should override ``calc_masks`` method to specify pruning strategy.

    Parameters
    ----------
    model : tf.keras.Model
        The user model to prune.
    config_list : list of JSON object
        User configuration. The format is detailed in tutorial.
    """
    def __init__(self, model, config_list):
        super().__init__(PrunerLayerWrapper, model, config_list)
        #self.callback = PrunerCallback(self)

    def compress(self):
        """
        Apply compression on a pre-trained model.

        If you want to prune the model during training, use callback API (WIP) instead.

        Returns
        -------
        tf.keras.Model
            The compressed model, for convenience. This is exactly the same object to constructor argument.
        """
        self._update_mask()
        return self.bound_model

    def calc_masks(self, wrapper, **kwargs):
        """
        Abstract method to be overridden by algorithm. End users should ignore it.

        If the callback is set up, this method will be invoked at end of each training minibatch.
        If not, it will only be called when end user invokes ``compress``.

        Parameters
        ----------
        wrapper : PrunerLayerWrapper
            The instrumented layer.
        **kwargs
            Reserved for forward compatibility.

        Returns
        -------
        dict of (str, tf.Tensor), or None
            The key is weight ``Variable``'s name. The value is a mask ``Tensor`` of weight's shape and dtype.
            If a weight's key does not appear in the return value, that weight will not be pruned.
            Returning ``None`` means the mask is not changed since last time.
            Weight names are globally unique, e.g. `model/conv_1/kernel:0`.
        """
        # TODO: maybe it should be able to calc on weight-granularity, beside from layer-granularity
        raise NotImplementedError("Pruners must overload calc_masks()")

    def _update_mask(self):
        for wrapper_idx, wrapper in enumerate(self.wrappers):
            masks = self.calc_masks(wrapper, wrapper_idx=wrapper_idx)
            if masks is not None:
                wrapper.masks = masks


class PrunerLayerWrapper(tf.keras.Model):
    """
    Instrumented TF layer.

    Wrappers will be passed to pruner's ``calc_masks`` API,
    and the pruning algorithm should use wrapper's attributes to calculate masks.

    Once instrumented, underlying layer's weights will get **modified** by masks before forward pass.

    Attributes
    ----------
    layer_info : LayerInfo
        All static information of the original layer.
    layer : tf.keras.layers.Layer
        The original layer.
    config : JSON object
        Selected configuration. The format is detailed in tutorial.
    pruner : Pruner
        Bound pruner object.
    masks : dict of (str, tf.Tensor)
        Current masks. The key is weight's name and the value is mask tensor.
        On initialization, `masks` is an empty dict, which means no weight is pruned.
        Afterwards, `masks` is the last return value of ``Pruner.calc_masks``.
        See ``Pruner.calc_masks`` for details.
    """
    def __init__(self, layer_info, pruner):
        super().__init__()
        self.layer_info = layer_info
        self.layer = layer_info.layer
        self.config = layer_info.config
        self.pruner = pruner
        self.masks = {}
        _logger.info('Layer detected to compress: %s', self.layer.name)

    def call(self, *inputs):
        new_weights = []
        for weight in self.layer.weights:
            mask = self.masks.get(weight.name)
            if mask is not None:
                new_weights.append(tf.math.multiply(weight, mask))
            else:
                new_weights.append(weight)
        if new_weights and not hasattr(new_weights[0], 'numpy'):
            raise RuntimeError('NNI: Compressed model can only run in eager mode')
        self.layer.set_weights([weight.numpy() for weight in new_weights])
        return self.layer(*inputs)


# TODO: designed to replace `patch_optimizer`
#class PrunerCallback(tf.keras.callbacks.Callback):
#    def __init__(self, pruner):
#        super().__init__()
#        self._pruner = pruner
#
#    def on_train_batch_end(self, batch, logs=None):
#        self._pruner.update_mask()


def _detect_layers_to_compress(model, config_list):
    # Returns list of LayerInfo.
    located_layers = _locate_layers(model)
    ret = []
    for layer in model.layers:
        config = _select_config(LayerInfo(layer), config_list)
        if config is not None:
            if id(layer) not in located_layers:
                _logger.error('Failed to locate layer %s in model. The layer will not be compressed. '
                              'This is a bug in NNI, feel free to fire an issue.', layer.name)
                continue
            layer_info = located_layers[id(layer)]
            layer_info.config = config
            ret.append(layer_info)
    return ret

def _locate_layers(model, cur_path=[]):
    # Find out how to access layers from model object.
    # Returns dict of (layer's object ID, LayerInfo).
    # This function is required because TF framework does not track layer's attribute name,
    # and to my knowledge `Layer.name` is only useful for read-only access.
    # `cur_path`s format is documented in `LayerInfo.path`.
    # TODO: it can only find layers in `Model` and `list` for now.
    assert isinstance(model, tf.keras.Model)
    if isinstance(model, tf.keras.Sequential):
        _logger.warning('`Sequential` model is not supported yet, ignored.')
    ret = {}
    for key, value in model.__dict__.items():
        if isinstance(value, tf.keras.Model):
            ret.update(_locate_layers(value, cur_path + [key]))
        elif isinstance(value, tf.keras.layers.Layer):
            ret[id(value)] = LayerInfo(value, cur_path + [key])
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, tf.keras.Model):
                    ret.update(_locate_layers(item, cur_path + [(key, i)]))
                elif isinstance(item, tf.keras.layers.Layer):
                    ret[id(item)] = LayerInfo(item, cur_path + [(key, i)])
    return ret

def _select_config(layer_info, config_list):
    # Find the last matching config block for given layer.
    # Returns None if the layer should not be compressed.
    ret = None
    for config in config_list:
        if 'op_types' in config:
            match = layer_info.type in config['op_types']
            match_default = 'default' in config['op_types'] and layer_info.type in default_layers.weighted_modules
            if not match and not match_default:
                continue
        if 'op_names' in config and layer_info.name not in config['op_names']:
            continue
        ret = config
    if ret is None or 'exclude' in ret:
        return None
    return ret


def _instrument_model(model, wrappers):
    # Replace layers to wrappers
    for wrapper in reversed(wrappers):
        cur = model
        for key in wrapper.layer_info.path[:-1]:
            if isinstance(key, str):
                cur = getattr(cur, key)
            else:
                name, index = key
                cur = getattr(cur, name)[index]
        key = wrapper.layer_info.path[-1]
        if isinstance(key, str):
            setattr(cur, key, wrapper)
        else:
            name, index = key
            getattr(cur, name)[index] = wrapper
            #if isinstance(cur, tf.keras.Sequential):
            #    cur._graph_initialized = False
            #    cur._layer_call_argspecs[wrapper] = cur._layer_call_argspecs[wrapper.layer]
