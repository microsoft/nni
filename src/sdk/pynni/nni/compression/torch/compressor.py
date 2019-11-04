import logging
import torch
from . import default_layers

_logger = logging.getLogger(__name__)


class LayerInfo:
    def __init__(self, name, module):
        self.module = module
        self.name = name
        self.type = type(module).__name__

        self._forward = None

    def save_forward(self):
        self._forward = self.module.forward

    def mount_forward(self, new_forward):
        self.module.forward = new_forward

class Compressor:
    """
    Abstract base PyTorch compressor
    """

    def __init__(self, model, config_list):
        """
        Record necessary info in class members

        Parameters
        ----------
        model : pytorch model
            the model user wants to compress
        config_list : list
            the configurations that users specify for compression
        """
        self.bound_model = model
        self.config_list = config_list
        self.modules_to_compress = None
        self.detect_modules_to_compress()

    def detect_modules_to_compress(self):
        """
        detect all modules should be compressed, and save the result in `self.modules_to_compress`.

        The model will be instrumented and user should never edit it after calling this method.
        """
        if self.modules_to_compress is None:
            self.modules_to_compress = []
            for name, module in self.bound_model.named_modules():
                layer = LayerInfo(name, module)
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
        for layer, config in self.modules_to_compress:
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
        layer : LayerInfo
            one layer

        Returns
        -------
        ret : config or None
            the retrieved configuration for this layer, if None, this layer should
            not be compressed
        """
        ret = None
        for config in self.config_list:
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
    Abstract base PyTorch pruner
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
        assert layer._forward is None, 'Each model can only be compressed once'
        if not _check_weight(layer.module):
            _logger.warning('Module %s does not have parameter "weight"', layer.name)
            return
        layer._forward = layer.module.forward

        def new_forward(*inputs):
            # apply mask to weight
            old_weight = layer.module.weight.data
            mask = self.calc_mask(layer, config)
            layer.module.weight.data = old_weight.mul(mask)
            # calculate forward
            ret = layer._forward(*inputs)
            return ret

        layer.module.forward = new_forward


class Quantizer(Compressor):
    """
    Base quantizer for pytorch quantizer
    """

    def quantize_weight(self, weight, config, op, op_type, op_name):
        """
        quantize should overload this method to quantize weight tensors.
        This method is effectively hooked to `forward()` method of the model.

        Parameters
        ----------
        weight : Tensor
            weight that needs to be quantized
        config : dict
            the configuration for weight quantization
        """
        raise NotImplementedError("Quantizer must overload quantize_weight()")

    def quantize_activation(self, activation, config, op, op_type, op_name):
        """
        quantize should overload this method to quantize activations.
        This method is effectively hooked to `forward()` method of the model.

        Parameters
        ----------
        activations : Tensor
            activations that needs to be quantized
        config : dict
            the configuration for activation quantization
        """
        raise NotImplementedError("Quantizer must overload quantize_activation()")

    def quantize_inputs(self, *inputs, config, op, op_type, op_name):
        """
        quantize should overload this method to quantize input.
        This method is effectively hooked to `forward()` method of the model.

        Parameters
        ----------
        input : Tensor
            activations that needs to be quantized
        config : dict
            the configuration for inputs quantization
        """
        raise NotImplementedError("Quantizer must overload quantize_inputs()")


    def _instrument_layer(self, layer, config):
        """
        Create a wrapper forward function to replace the original one.

        Parameters
        ----------
        layer : LayerInfo
            the layer to instrument the mask
        config : dict
            the configuration for quantization
        """
        assert layer._forward is None, 'Each model can only be compressed once'
        if config.get("weight_quantization", False):
            if not _check_weight(layer.module):
                _logger.warning('Module %s does not have parameter "weight"', layer.name)
                return
        layer.save_forward()

        def new_forward(*inputs):
            if config.get("input_quantization", False):
                inputs = self.quantize_inputs(inputs, config=config, op=layer.module, op_type=layer.type, op_name=layer.name)

            if config.get("weight_quantization", False):
                weight = layer.module.weight.data
                new_weight = self.quantize_weight(weight, config, op=layer.module, op_type=layer.type, op_name=layer.name)
                layer.module.weight.data = new_weight
                result = layer._forward(*inputs)
                layer.module.weight.data = weight
            else:
                result = layer._forward(*inputs)

            if config.get("activation_quantization", False):
                result = self.quantize_activation(result, config, op=layer.module, op_type=layer.type, op_name=layer.name)

            return result

        layer.mount_forward(new_forward)

def _check_weight(module):
    try:
        return isinstance(module.weight, torch.nn.Parameter) and isinstance(module.weight.data, torch.Tensor)
    except AttributeError:
        return False
