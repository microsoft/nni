from torch import Tensor
from torch.nn import Module, Parameter
from ruamel.yaml import YAML
from typing import List
import logging

logger = logging.getLogger('torch_compressor')
__all__ = [
    'TorchCompressor',
    'TorchPruner',
    'TorchQuantizer',
    '_torch_detect_prunable_layers',
    '_torch_default_get_configure',
    '_torch_default_load_configure_file'
]


class TorchCompressor:
    """
    Base compressor for pytorch
    """

    def __init__(self):
        self._bound_model = None


    def compress(self, model):
        """
        Compress the model with algorithm implemented by subclass.
        The model will be instrumented and user should never edit it after calling this method.
        """
        assert self._bound_model is None, "Each NNI compressor instance can only compress one model"
        self._bound_model = model
        self.preprocess_model(model)


    def preprocess_model(self, model):
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


class TorchLayerInfo:
    """
    layer info for pytorch
    """
    def __init__(self, name, layer):
        self.name = name
        self.layer = layer

        self._forward = None


def _torch_detect_prunable_layers(model):
    # search for all layers which have parameter "weight"
    ret = []
    for name, layer in model.named_modules():
        try:
            if isinstance(layer.weight, Parameter) and isinstance(layer.weight.data, Tensor):
                ret.append(TorchLayerInfo(name, layer))
        except AttributeError:
            pass
    return ret

def _torch_default_get_configure(configure_list, layer_info):
    """
    Get configure for input layer
    defaultly the later config will cover front config
    WARNING: please mask sure default configure is the first in the list
    """
    if not configure_list:
        logger.warning('WARNING: configure list is None')
    configure = {}
    for config in configure_list:
        if config.get('support_type', '') == 'default':
            configure = config
        elif type(layer_info.layer).__name__ in config.get('support_type', []):
            configure = config
        elif layer_info.name in config.get('support_op', []):
            configure = config
    if not configure:
        logger.warning('WARNING: can not get configure, default NONE!!!')
    return configure

def _torch_default_load_configure_file(config_path, class_name):
    logger.info('load CLASS:{0} from PATH:{1}'.format(class_name, config_path))
    assert config_path is not None and config_path.endswith('yaml')
    file = open(config_path, 'r')
    yaml = YAML(typ='safe')
    yaml_text = yaml.load(file.read())
    configure_file = yaml_text.get(class_name, {})
    if not configure_file:
        logger.warning('WARNING: load Nothing from configure file, Default { }')
    return configure_file


class TorchPruner(TorchCompressor):
    """
    Base pruner for pytorch pruner
    """

    def __init__(self):
        super().__init__()

    def __call__(self, model):
        self.compress(model)
        return model

    def calc_mask(self, layer_info, weight):
        """
        Pruners should overload this method to provide mask for weight tensors.
        The mask must have the same shape and type comparing to the weight.
        It will be applied with `mul()` operation.
        This method is effectively hooked to `forward()` method of the model.
        """
        raise NotImplementedError("Pruners must overload calc_mask()")


    def compress(self, model):
        super().compress(model)
        # TODO: configurable whitelist
        for layer_info in _torch_detect_prunable_layers(model):
            self._instrument_layer(layer_info)

    def _instrument_layer(self, layer_info):
        # TODO: bind additional properties to layer_info instead of layer
        # create a wrapper forward function to replace the original one
        assert layer_info._forward is None, 'Each model can only be compressed once'
        layer_info._forward = layer_info.layer.forward

        def new_forward(*input):
            # apply mask to weight
            mask = self.calc_mask(layer_info, layer_info.layer.weight.data)
            layer_info._backup_weight = layer_info.layer.weight.data
            layer_info.layer.weight.data = layer_info.layer.weight.data.mul(mask)
            # calculate forward
            ret = layer_info._forward(*input)
            # recover original weight
            layer_info.layer.weight.data = layer_info._backup_weight
            return ret

        layer_info.layer.forward = new_forward
    


class TorchQuantizer(TorchCompressor):
    """
    Base quantizer for pytorch quantizer
    """
    def __init__(self):
        super().__init__()

    def __call__(self, model):
        self.compress(model)
        return model
    
    def quantize_weight(self, layer_info, weight):
        """
        user should know where dequantize goes and implement it in quantize method
        we now do not provide dequantize method
        """
        raise NotImplementedError("Quantizer must overload quantize_weight()")

    def compress(self, model):
        super().compress(model)
        count = 0
        for layer_info in _torch_detect_prunable_layers(model):
            if count == 0:
                count = count +1
                continue
            self._instrument_layer(layer_info)

    def _instrument_layer(self, layer_info):
        assert layer_info._forward is None
        layer_info._forward = layer_info.layer.forward

        def new_forward(*input):
            layer_info.layer.weight.data = self.quantize_weight(layer_info, layer_info.layer.weight.data)
            return layer_info._forward(*input)

        layer_info.layer.forward = new_forward
    
    
