# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import tensorflow as tf
from . import default_layers
tf.config.experimental_run_functions_eagerly(True)

_logger = logging.getLogger(__name__)


class LayerInfo:
    def __init__(self, keras_layer):
        self.keras_layer = keras_layer
        self.name = keras_layer.name
        self.type = default_layers.get_op_type(type(keras_layer))
        self.weight_index = default_layers.get_weight_index(self.type)
        if self.weight_index is not None:
            self.weight = keras_layer.weights[self.weight_index]
        self._call = None

class Compressor:
    """
    Abstract base TensorFlow compressor
    """

    def __init__(self, model, config_list):
        """
        Record necessary info in class members

        Parameters
        ----------
        model : keras model
            the model user wants to compress
        config_list : list
            the configurations that users specify for compression
        """
        self.bound_model = model
        self.config_list = config_list
        self.modules_to_compress = []

    def detect_modules_to_compress(self):
        """
        detect all modules should be compressed, and save the result in `self.modules_to_compress`.

        The model will be instrumented and user should never edit it after calling this method.
        """
        if self.modules_to_compress is None:
            self.modules_to_compress = []
            for keras_layer in self.bound_model.layers:
                layer = LayerInfo(keras_layer)
                config = self.select_config(layer)
                if config is not None:
                    self.modules_to_compress.append((layer, config))
        return self.modules_to_compress

    def compress(self):
        """
        Compress the model with algorithm implemented by subclass.

        The model will be instrumented and user should never edit it after calling this method.
        `self.modules_to_compress` records all the to-be-compressed layers
        """
        modules_to_compress = self.detect_modules_to_compress()
        for layer, config in modules_to_compress:
            self._instrument_layer(layer, config)
        return self.bound_model

    def get_modules_to_compress(self):
        """
        To obtain all the to-be-compressed layers.

        Returns
        -------
        self.modules_to_compress : list
            a list of the layers, each of which is a tuple (`layer`, `config`),
            `layer` is `LayerInfo`, `config` is a `dict`
        """
        return self.modules_to_compress

    def select_config(self, layer):
        """
        Find the configuration for `layer` by parsing `self.config_list`

        Parameters
        ----------
        layer: LayerInfo
            one layer

        Returns
        -------
        ret : config or None
            the retrieved configuration for this layer, if None, this layer should
            not be compressed
        """
        ret = None
        if layer.type is None:
            return None
        for config in self.config_list:
            config = config.copy()
            config['op_types'] = self._expand_config_op_types(config)
            if layer.type not in config['op_types']:
                continue
            if config.get('op_names') and layer.name not in config['op_names']:
                continue
            ret = config
        if ret is None or ret.get('exclude'):
            return None
        return ret

    def update_epoch(self, epoch):
        """
        If user want to update model every epoch, user can override this method.
        This method should be called at the beginning of each epoch

        Parameters
        ----------
        epoch : num
            the current epoch number
        """

    def step(self):
        """
        If user want to update mask every step, user can override this method
        """


    def _instrument_layer(self, layer, config):
        """
        This method is implemented in the subclasses, i.e., `Pruner` and `Quantizer`

        Parameters
        ----------
        layer : LayerInfo
            the layer to instrument the compression operation
        config : dict
            the configuration for compressing this layer
        """
        raise NotImplementedError()

    def _expand_config_op_types(self, config):
        if config is None:
            return []
        op_types = []

        for op_type in config.get('op_types', []):
            if op_type == 'default':
                op_types.extend(default_layers.default_layers)
            else:
                op_types.append(op_type)
        return op_types


class Pruner(Compressor):
    """
    Abstract base TensorFlow pruner
    """

    def calc_mask(self, layer, config):
        """
        Pruners should overload this method to provide mask for weight tensors.
        The mask must have the same shape and type comparing to the weight.
        It will be applied with `mul()` operation on the weight.
        This method is effectively hooked to `forward()` method of the model.

        Parameters
        ----------
        layer : LayerInfo
            calculate mask for `layer`'s weight
        config : dict
            the configuration for generating the mask
        """
        raise NotImplementedError("Pruners must overload calc_mask()")

    def _instrument_layer(self, layer, config):
        """
        Create a wrapper forward function to replace the original one.

        Parameters
        ----------
        layer : LayerInfo
            the layer to instrument the mask
        config : dict
            the configuration for generating the mask
        """
        layer._call = layer.keras_layer.call

        def new_call(*inputs):
            weights = [x.numpy() for x in layer.keras_layer.weights]
            mask = self.calc_mask(layer, config)
            weights[layer.weight_index] = weights[layer.weight_index] * mask
            layer.keras_layer.set_weights(weights)
            ret = layer._call(*inputs)
            return ret

        layer.keras_layer.call = new_call

class Quantizer(Compressor):
    """
    Abstract base TensorFlow quantizer
    """

    def quantize_weight(self, weight, config, op, op_type, op_name):
        raise NotImplementedError("Quantizer must overload quantize_weight()")
