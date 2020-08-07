# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import tensorflow as tf
from . import default_layers

_logger = logging.getLogger(__name__)


class LayerInfo:
    def __init__(self, layer, path=None):
        self.layer = layer
        self.name = layer.name
        self.type = type(layer).__name__
        self.path = path
        self.config = None


class Compressor:
    def __init__(self, LayerWrapperClass, model, config_list):
        assert isinstance(model, tf.keras.Model)
        self.validate_config(model, config_list)

        self.bound_model = model
        self.wrappers = []

        for layer_info in _detect_layers_to_compress(model, config_list):
            self.wrappers.append(LayerWrapperClass(layer_info, self))
        if not self.wrappers:
            _logger.warning('Nothing is configured to compress, please check your model and config list')

        _instrument_model(model, self.wrappers)

    def set_wrappers_attribute(self, name, value):
        for wrapper in self.wrappers:
            setattr(wrapper, name, value)


class Pruner(Compressor):
    def __init__(self, model, config_list):
        super().__init__(PrunerLayerWrapper, model, config_list)
        #self.callback = PrunerCallback(self)

    def compress(self):
        self.update_mask()
        return self.bound_model

    def update_mask(self):
        for wrapper_idx, wrapper in enumerate(self.wrappers):
            masks = self.calc_masks(wrapper, wrapper_idx=wrapper_idx)
            if masks is not None:
                wrapper.masks = masks

    def calc_masks(self, wrapper, **kwargs):
        # TODO: maybe it should be able to calc on weight-granularity, beside from layer-granularity
        raise NotImplementedError("Pruners must overload calc_masks()")


class PrunerLayerWrapper(tf.keras.Model):
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
                new_weights.append(tf.math.multiply(weight, mask).numpy())
            else:
                new_weights.append(weight.numpy())
        self.layer.set_weights(new_weights)
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
    # FIXME: this cannot find layers contained in list, dict, non-model custom classes, etc
    ret = {}

    if isinstance(model, tf.keras.Model):
        for key, value in model.__dict__.items():
            if isinstance(value, tf.keras.Model):
                ret.update(_locate_layers(value, cur_path + [key]))
            elif isinstance(value, list):
                ret.update(_locate_layers(value, cur_path + [key]))
            elif isinstance(value, tf.keras.layers.Layer):
                ret[id(value)] = LayerInfo(value, cur_path + [key])

    elif isinstance(model, list):
        for i, item in enumerate(model):
            if isinstance(item, tf.keras.Model):
                ret.update(_locate_layers(item, cur_path + [i]))
            elif isinstance(item, tf.keras.layers.Layer):
                ret[id(item)] = LayerInfo(item, cur_path + [i])

    else:
        raise ValueError('Unexpected model type: {}'.format(type(model)))
    return ret

def _select_config(layer_info, config_list):
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
    for wrapper in wrappers:
        cur = model
        for key in wrapper.layer_info.path[:-1]:
            if isinstance(key, int):
                cur = cur[key]
            else:
                cur = getattr(cur, key)
        key = wrapper.layer_info.path[-1]
        if isinstance(key, int):
            cur[key] = wrapper
        else:
            setattr(cur, key, wrapper)
