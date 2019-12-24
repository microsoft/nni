# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

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


class Quantizer(Compressor):
    """
    Base quantizer for pytorch quantizer
    """

    def __init__(self, model, config_list):
        super().__init__(model, config_list)
        self.quant_grad = QuantGrad

    def quantize_weight(self, weight, config, op, op_type, op_name):
        """
        quantize should overload this method to quantize weight.
        This method is effectively hooked to :meth:`forward` of the model.
        Parameters
        ----------
        weight : Tensor
            weight that needs to be quantized
        config : dict
            the configuration for weight quantization
        """
        raise NotImplementedError('Quantizer must overload quantize_weight()')

    def quantize_output(self, output, config, op, op_type, op_name):
        """
        quantize should overload this method to quantize output.
        This method is effectively hooked to :meth:`forward` of the model.
        Parameters
        ----------
        output : Tensor
            output that needs to be quantized
        config : dict
            the configuration for output quantization
        """
        raise NotImplementedError('Quantizer must overload quantize_output()')

    def quantize_input(self, *inputs, config, op, op_type, op_name):
        """
        quantize should overload this method to quantize input.
        This method is effectively hooked to :meth:`forward` of the model.
        Parameters
        ----------
        inputs : Tensor
            inputs that needs to be quantized
        config : dict
            the configuration for inputs quantization
        """
        raise NotImplementedError('Quantizer must overload quantize_input()')


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
        assert 'quant_types' in config, 'must provide quant_types in config'
        assert isinstance(config['quant_types'], list), 'quant_types must be list type'
        assert 'quant_bits' in config, 'must provide quant_bits in config'
        assert isinstance(config['quant_bits'], int) or isinstance(config['quant_bits'], dict), 'quant_bits must be dict type or int type'

        if isinstance(config['quant_bits'], dict):
            for quant_type in config['quant_types']:
                assert quant_type in config['quant_bits'], 'bits length for %s must be specified in quant_bits dict' % quant_type

        if 'weight' in config['quant_types']:
            if not _check_weight(layer.module):
                _logger.warning('Module %s does not have parameter "weight"', layer.name)
            else:
                # old_weight is used to store origin weight and weight is used to store quantized weight
                # the reason why weight is buffer instead of parameter is because in pytorch parameter is used as leaf
                # if weight is leaf , then old_weight can not be updated.
                layer.module.register_parameter('old_weight', torch.nn.Parameter(layer.module.weight))
                delattr(layer.module, 'weight')
                layer.module.register_buffer('weight', layer.module.old_weight)

        layer._forward = layer.module.forward

        def new_forward(*inputs):
            if 'input' in config['quant_types']:
                inputs = self.quant_grad.apply(inputs, QuantType.QUANT_INPUT, self.quantize_input, config, layer)

            if 'weight' in config['quant_types'] and _check_weight(layer.module):
                new_weight = self.quant_grad.apply(layer.module.old_weight, QuantType.QUANT_WEIGHT, self.quantize_weight, config, layer)
                layer.module.weight = new_weight
                result = layer._forward(*inputs)
            else:
                result = layer._forward(*inputs)

            if 'output' in config['quant_types']:
                result = self.quant_grad.apply(result, QuantType.QUANT_OUTPUT, self.quantize_output, config, layer)
            return result

        layer.module.forward = new_forward

class QuantType:
    """
    Enum class for quantization type.
    """
    QUANT_INPUT = 0
    QUANT_WEIGHT = 1
    QUANT_OUTPUT = 2

class QuantGrad(torch.autograd.Function):
    """
    Base class for overriding backward function of quantization operation.
    """
    @staticmethod
    def quant_backward(tensor, grad_output, quant_type):
        """
        This method should be overrided by subclass to provide customized backward function,
        default implementation is Straight-Through Estimator
        Parameters
        ----------
        tensor : Tensor
            input of quantization operation
        grad_output : Tensor
            gradient of the output of quantization operation
        quant_type : QuantType
            the type of quantization, it can be `QuantType.QUANT_INPUT`, `QuantType.QUANT_WEIGHT`, `QuantType.QUANT_OUTPUT`,
            you can define different behavior for different types.
        Returns
        -------
        tensor
            gradient of the input of quantization operation
        """
        return grad_output

    @staticmethod
    def forward(ctx, tensor, quant_type, quant_func, config, layer):
        ctx.save_for_backward(tensor, torch.Tensor([quant_type]))
        return quant_func(tensor, config, op=layer.module, op_type=layer.type, op_name=layer.name)

    @classmethod
    def backward(cls, ctx, grad_output):
        tensor, quant_type = ctx.saved_variables
        output = cls.quant_backward(tensor, grad_output, quant_type)
        return output, None, None, None, None

def _check_weight(module):
    try:
        return isinstance(module.weight.data, torch.Tensor)
    except AttributeError:
        return False
    