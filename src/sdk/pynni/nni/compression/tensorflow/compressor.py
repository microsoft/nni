# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import tensorflow as tf
from . import default_layers

_logger = logging.getLogger(__name__)


class LayerInfo:
    def __init__(self, layer):
        self.module = layer
        self.name = layer._name
        self.type = type(layer).__name__


def _wrap_model(model, wrapped_layers):
    for key, value in model.__dict__.items():
        if isinstance(value, tf.keras.Model):
            _wrap_model(value, wrapped_layers)
        for layer in wrapped_layers:
            if value is layer.module:
                setattr(model, key, layer)


class Compressor:
    def __init__(self, model, config_list, optimizer=None):
        assert isinstance(model, tf.keras.Model)
        self.validate_config(model, config_list)

        self.bound_model = model
        self.config_list = config_list
        self.optimizer = optimizer

        self.modules_to_compress = None
        self.modules_wrapper = []

        for layer, config in self._detect_modules_to_compress():
            wrapper = self._wrap_modules(layer, config)
            self.modules_wrapper.append(wrapper)
        if not self.modules_wrapper:
            _logger.warning('Nothing is configured to compress, please check your model and config list')

        _wrap_model(model, self.modules_wrapper)

    def _detect_modules_to_compress(self):
        if self.modules_to_compress is None:
            self.modules_to_compress = []
            for keras_layer in self.bound_model.layers:
                layer = LayerInfo(keras_layer)
                config = self.select_config(layer)
                if config is not None:
                    self.modules_to_compress.append((layer, config))
        return self.modules_to_compress

    def compress(self):
        return self.bound_model

    def set_wrappers_attribute(self, name, value):
        for wrapper in self.get_modules_wrapper():
            setattr(wrapper, name, value)

    def get_modules_to_compress(self):
        return self.modules_to_compress

    def select_config(self, layer):
        ret = None
        if layer.type is None:
            return None
        for config in self.config_list:
            config = config.copy()
            if 'op_types' in config and 'default' in config['op_type']:
                expanded_op_types = []
            for op_type in config['op_types']:
                if op_type == 'default':
                    expanded_op_types.extend(default_layers.weighted_modules)
                else:
                    expanded_op_types.append(op_type)
            config['op_types'] = expanded_op_types

            if 'op_types' in config and layer.type not in config['op_types']:
                continue
            if 'op_names' in config and layer.name not in config['op_names']:
                continue

            ret = config

        if ret is None or 'exclude' is ret:
            return None
        return ret


    def update_epoch(self, epoch):
        pass


    def _wrap_modules(self, layer, config):
        raise NotImplementedError()


    def patch_optimizer(self, **tasks):
        pass


class Pruner(Compressor):
    def __init__(self, model, config_list, optimizer=None):
        super().__init__(model, config_list, optimizer)
        if optimizer is not None:
            self.patch_optimizer(self.update_mask)

    def compress(self):
        self.update_mask()
        return self.bound_model

    def update_mask(self):
        for wrapper_idx, wrapper in enumerate(self.get_modules_wrapper()):
            masks = self.calc_mask(wrapper, wrapper_idx=wrapper_idx)
            if masks is not None:
                for k in masks:
                    assert hasattr(wrapper, k)
                    setattr(wrapper, k, masks[k])

    def calc_mask(self, wrapper, **kwargs):
        raise NotImplementedError("Pruners must overload calc_mask()")

    def _wrap_modules(self, layer, config):
        _logger.info('Module detected to compress : %s.', layer.name)
        return PrunerModuleWrapper(layer.module, layer.name, layer.type, config, self)


class PrunerModuleWrapper(tf.keras.Model):
    def __init__(self, module, module_name, module_type, config, pruner):
        super().__init__()
        self.module = module
        self.name = module_name
        self.type = module_type
        self.config = config
        self.pruner = pruner
        self.masks = []
        for weight in module.weights:
            self.masks.append(tf.ones_like(weight))
            # TODO: filter weight name like 'kernel'/'bias'/etc?

    def call(self, *inputs):
        new_weights = []
        for mask, weight in zip(self.masks, self.module.weights):
            new_weights.append(tf.math.multiply(mask, weight).numpy())
        self.module.set_weights(new_weights)
        return self.module(*inputs)
