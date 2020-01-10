# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import onnx
import torch
from . import default_layers

_logger = logging.getLogger(__name__)


class LayerInfo:
    def __init__(self, name, module):
        self.module = module
        self.name = name
        self.type = type(module).__name__

        self._forward = None


class ModelSpeedup:
    """
    Abstract base PyTorch ModelSpeedup
    """

    def __init__(self, model, dummy_input, masks_file):
        """
        Record necessary info in class members

        Parameters
        ----------
        model : pytorch model
            the model user wants to compress
        masks : dict
            the generated masks for modules,
            key is module name,
            value is a dict including key `weight`, or also key `bias`
        onnx_graph : xxx
            it is used to parse dependencies between modules
        """
        self.bound_model = model
        self.masks = torch.load(masks_file)
        self.model_graph = torch.jit.trace(model, dummy_input)

    def expand_masks(self):
        """
        """
        for name, mask in self.masks:
            print(name)

        if self.modules_to_compress is None:
            self.modules_to_compress = []
            for name, module in self.bound_model.named_modules():
                layer = LayerInfo(name, module)
                config = self.select_config(layer)
                if config is not None:
                    self.modules_to_compress.append((layer, config))
        return self.modules_to_compress

    def compress_modules(self):
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
        list
            a list of the layers, each of which is a tuple (`layer`, `config`),
            `layer` is `LayerInfo`, `config` is a `dict`
        """
        return self.modules_to_compress

    def select_config(self, layer):
        """
        Find the configuration for `layer` by parsing `self.config_list`

        Parameters
        ----------
        layer : LayerInfo
            one layer

        Returns
        -------
        config or None
            the retrieved configuration for this layer, if None, this layer should
            not be compressed
        """
        ret = None
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
        If user want to update model every step, user can override this method
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
        expanded_op_types = []
        for op_type in config.get('op_types', []):
            if op_type == 'default':
                expanded_op_types.extend(default_layers.weighted_modules)
            else:
                expanded_op_types.append(op_type)
        return expanded_op_types


class Pruner(Compressor):
    """
    Prune to an exact pruning level specification

    Attributes
    ----------
    mask_dict : dict
        Dictionary for saving masks, `key` should be layer name and
        `value` should be a tensor which has the same shape with layer's weight

    """

    def __init__(self, model, config_list):
        super().__init__(model, config_list)
        self.mask_dict = {}

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
        assert layer._forward is None, 'Each model can only be compressed once'
        if not _check_weight(layer.module):
            _logger.warning('Module %s does not have parameter "weight"', layer.name)
            return
        layer._forward = layer.module.forward

        def new_forward(*inputs):
            mask = self.calc_mask(layer, config)
            # apply mask to weight
            old_weight = layer.module.weight.data
            mask_weight = mask['weight']
            layer.module.weight.data = old_weight.mul(mask_weight)
            # apply mask to bias
            if mask.__contains__('bias') and hasattr(layer.module, 'bias') and layer.module.bias is not None:
                old_bias = layer.module.bias.data
                mask_bias = mask['bias']
                layer.module.bias.data = old_bias.mul(mask_bias)
            # calculate forward
            ret = layer._forward(*inputs)
            return ret

        layer.module.forward = new_forward

    def export_model(self, model_path, mask_path=None, onnx_path=None, input_shape=None):
        """
        Export pruned model weights, masks and onnx model(optional)

        Parameters
        ----------
        model_path : str
            path to save pruned model state_dict
        mask_path : str
            (optional) path to save mask dict
        onnx_path : str
            (optional) path to save onnx model
        input_shape : list or tuple
            input shape to onnx model
        """
        if self.detect_modules_to_compress() and not self.mask_dict:
            _logger.warning('You may not use self.mask_dict in base Pruner class to record masks')
        assert model_path is not None, 'model_path must be specified'
        for name, m in self.bound_model.named_modules():
            if name == "":
                continue
            masks = self.mask_dict.get(name)
            if masks is not None:
                mask_sum = masks['weight'].sum().item()
                mask_num = masks['weight'].numel()
                _logger.info('Layer: %s  Sparsity: %.2f', name, 1 - mask_sum / mask_num)
                m.weight.data = m.weight.data.mul(masks['weight'])
                if masks.__contains__('bias') and hasattr(m, 'bias') and m.bias is not None:
                    m.bias.data = m.bias.data.mul(masks['bias'])
            else:
                _logger.info('Layer: %s  NOT compressed', name)
        torch.save(self.bound_model.state_dict(), model_path)
        _logger.info('Model state_dict saved to %s', model_path)
        if mask_path is not None:
            torch.save(self.mask_dict, mask_path)
            _logger.info('Mask dict saved to %s', mask_path)
        if onnx_path is not None:
            assert input_shape is not None, 'input_shape must be specified to export onnx model'
            # input info needed
            input_data = torch.Tensor(*input_shape)
            torch.onnx.export(self.bound_model, input_data, onnx_path)
            _logger.info('Model in onnx with input shape %s saved to %s', input_data.shape, onnx_path)
