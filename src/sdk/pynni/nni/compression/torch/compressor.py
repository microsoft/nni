import torch
import logging
from . import default_layers

_logger = logging.getLogger(__name__)


class LayerInfo:
    def __init__(self, name, module):
        self.module = module
        self.name = name
        self.type = type(module).__name__

        self._forward = None
        self._backup_weight = None


class Compressor:
    """
    Abstract base PyTorch compressor
    """
    def __init__(self, config_list):
        self._bound_model = None
        self._config_list = config_list

    def __call__(self, model):
        self.compress(model)
        return model

    def compress(self, model):
        """
        Compress the model with algorithm implemented by subclass.
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
        """
        This method is called when a model is bound to the compressor.
        Users can optionally overload this method to do model-specific initialization.
        It is guaranteed that only one model will be bound to each compressor instance.
        """
        pass
    
    def update_epoch(self, epoch):
        """
        if user want to update model every epoch, user can override this method
        """
        pass
    
    def step(self):
        """
        if user want to update model every step, user can override this method
        """
        pass


    def _instrument_layer(self, layer, config):
        raise NotImplementedError()

    def _select_config(self, layer):
        ret = None
        for config in self._config_list:
            op_type = config.get('op_type')
            if op_type == 'default':
                op_type = default_layers.weighted_modules
            if op_type and layer.type not in op_type:
                continue
            if config.get('op_name') and layer.name not in config['op_name']:
                continue
            ret = config
        if ret is None or ret.get('exclude'):
            return None
        return ret


class Pruner(Compressor):
    """
    Abstract base PyTorch pruner
    """
    def __init__(self, config_list):
        super().__init__(config_list)

    def calc_mask(self, layer, weight, config):
        """
        Pruners should overload this method to provide mask for weight tensors.
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
            _logger.warning('Module {} does not have parameter "weight"'.format(layer.name))
            return
        layer._forward = layer.module.forward

        def new_forward(*input):
            # apply mask to weight
            mask = self.calc_mask(layer, layer.module.weight.data, config)
            layer._backup_weight = layer.module.weight.data
            layer.module.weight.data = layer.module.weight.data.mul(mask)
            # calculate forward
            ret = layer._forward(*input)
            # recover original weight
            layer.module.weight.data = layer._backup_weight
            return ret

        layer.module.forward = new_forward
 

class Quantizer(Compressor):
    """
    Base quantizer for pytorch quantizer
    """
    def __init__(self, config_list):
        super().__init__(config_list)

    def __call__(self, model):
        self.compress(model)
        return model
    
    def quantize_weight(self, layer, weight, config):
        """
        user should know where dequantize goes and implement it in quantize method
        we now do not provide dequantize method
        """
        raise NotImplementedError("Quantizer must overload quantize_weight()")

    def _instrument_layer(self, layer, config):
        assert layer._forward is None, 'Each model can only be compressed once'
        if not _check_weight(layer.module):
            _logger.warning('Module {} does not have parameter "weight"'.format(layer.name))
            return
        layer._forward = layer.module.forward

        def new_forward(*input):
            layer.module.weight.data = self.quantize_weight(layer, layer.module.weight.data, config)
            return layer._forward(*input)

        layer.module.forward = new_forward


def _check_weight(module):
    try:
        return isinstance(module.weight, torch.nn.Parameter) and isinstance(module.weight.data, torch.Tensor)
    except AttributeError:
        return False
