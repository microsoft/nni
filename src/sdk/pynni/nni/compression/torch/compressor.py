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


class Compressor:
    """Abstract base PyTorch compressor"""

    def __init__(self, config_list):
        self._bound_model = None
        self._config_list = config_list

    def __call__(self, model):
        self.compress(model)
        return model

    def compress(self, model):
        """Compress the model with algorithm implemented by subclass.
        The model will be instrumented and user should never edit it after calling this method.
        """
        assert self._bound_model is None, "Each NNI compressor instance can only compress one model"
        self._bound_model = model
        self.bind_model(model)
        for name, module in model.named_modules():
            layer = LayerInfo(name, module)
            config = self._select_config(layer)
            if config is not None:
                self._instrument_layer(layer, config)

    def bind_model(self, model):
        """This method is called when a model is bound to the compressor.
        Users can optionally overload this method to do model-specific initialization.
        It is guaranteed that only one model will be bound to each compressor instance.
        """

    def update_epoch(self, epoch):
        """if user want to update model every epoch, user can override this method
        """

    def step(self):
        """if user want to update model every step, user can override this method
        """

    def _instrument_layer(self, layer, config):
        raise NotImplementedError()

    def _select_config(self, layer):
        ret = None
        for config in self._config_list:
            op_types = config.get('op_types')
            if op_types == 'default':
                op_types = default_layers.weighted_modules
            if op_types and layer.type not in op_types:
                continue
            if config.get('op_names') and layer.name not in config['op_names']:
                continue
            ret = config
        if ret is None or ret.get('exclude'):
            return None
        return ret


class Pruner(Compressor):
    """
    Abstract base PyTorch pruner
    """

    def calc_mask(self, weight, config, op, op_type, op_name):
        """Pruners should overload this method to provide mask for weight tensors.
        The mask must have the same shape and type comparing to the weight.
        It will be applied with `mul()` operation.
        This method is effectively hooked to `forward()` method of the model.
        """
        raise NotImplementedError("Pruners must overload calc_mask()")

    def _instrument_layer(self, layer, config):
        # TODO: support multiple weight tensors
        # create a wrapper forward function to replace the original one
        assert layer._forward is None, 'Each model can only be compressed once'
        if not _check_weight(layer.module):
            _logger.warning('Module %s does not have parameter "weight"', layer.name)
            return
        layer._forward = layer.module.forward

        def new_forward(*inputs):
            # apply mask to weight
            old_weight = layer.module.weight.data
            mask = self.calc_mask(old_weight, config, op=layer.module, op_type=layer.type, op_name=layer.name)
            layer.module.weight.data = old_weight.mul(mask)
            # calculate forward
            ret = layer._forward(*inputs)
            # recover original weight
            layer.module.weight.data = old_weight
            return ret

        layer.module.forward = new_forward


class Quantizer(Compressor):
    """
    Base quantizer for pytorch quantizer
    """

    def __call__(self, model):
        self.compress(model)
        return model

    def quantize_weight(self, weight, config, op, op_type, op_name):
        """user should know where dequantize goes and implement it in quantize method
        we now do not provide dequantize method
        """
        raise NotImplementedError("Quantizer must overload quantize_weight()")

    def _instrument_layer(self, layer, config):
        assert layer._forward is None, 'Each model can only be compressed once'
        if not _check_weight(layer.module):
            _logger.warning('Module %s does not have parameter "weight"', layer.name)
            return
        layer._forward = layer.module.forward

        def new_forward(*inputs):
            weight = layer.module.weight.data
            new_weight = self.quantize_weight(weight, config, op=layer.module, op_type=layer.type, op_name=layer.name)
            layer.module.weight.data = new_weight
            return layer._forward(*inputs)

        layer.module.forward = new_forward


def _check_weight(module):
    try:
        return isinstance(module.weight, torch.nn.Parameter) and isinstance(module.weight.data, torch.Tensor)
    except AttributeError:
        return False
