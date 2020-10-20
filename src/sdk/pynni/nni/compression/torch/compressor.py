# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import types
import logging
import torch
from . import default_layers

_logger = logging.getLogger(__name__)


class LayerInfo:
    def __init__(self, name, module):
        self.module = module
        self.name = name
        self.type = type(module).__name__

def _setattr(model, name, module):
    name_list = name.split(".")
    for name in name_list[:-1]:
        model = getattr(model, name)
    setattr(model, name_list[-1], module)


class Compressor:
    """
    Abstract base PyTorch compressor
    """

    def __init__(self, model, config_list, optimizer=None):
        """
        Record necessary info in class members

        Parameters
        ----------
        model : pytorch model
            the model user wants to compress
        config_list : list
            the configurations that users specify for compression
        optimizer: pytorch optimizer
            optimizer used to train the model
        """
        assert isinstance(model, torch.nn.Module)
        self.validate_config(model, config_list)

        self.bound_model = model
        self.config_list = config_list
        self.optimizer = optimizer

        self.modules_to_compress = None
        self.modules_wrapper = []
        self.is_wrapped = False

        self._fwd_hook_handles = {}
        self._fwd_hook_id = 0

        self.reset()

        if not self.modules_wrapper:
            _logger.warning('Nothing is configured to compress, please check your model and config_list')

    def validate_config(self, model, config_list):
        """
        subclass can optionally implement this method to check if config_list if valid
        """
        pass

    def reset(self, checkpoint=None):
        """
        reset model state dict and model wrapper
        """
        self._unwrap_model()
        if checkpoint is not None:
            self.bound_model.load_state_dict(checkpoint)

        self.modules_to_compress = None
        self.modules_wrapper = []

        for layer, config in self._detect_modules_to_compress():
            wrapper = self._wrap_modules(layer, config)
            self.modules_wrapper.append(wrapper)

        self._wrap_model()

    def _detect_modules_to_compress(self):
        """
        detect all modules should be compressed, and save the result in `self.modules_to_compress`.
        The model will be instrumented and user should never edit it after calling this method.
        """
        if self.modules_to_compress is None:
            self.modules_to_compress = []
            for name, module in self.bound_model.named_modules():
                if module == self.bound_model:
                    continue
                layer = LayerInfo(name, module)
                config = self.select_config(layer)
                if config is not None:
                    self.modules_to_compress.append((layer, config))
        return self.modules_to_compress

    def _wrap_model(self):
        """
        wrap all modules that needed to be compressed

        """
        for wrapper in reversed(self.get_modules_wrapper()):
            _setattr(self.bound_model, wrapper.name, wrapper)
        self.is_wrapped = True

    def _unwrap_model(self):
        """
        unwrap all modules that needed to be compressed

        """
        for wrapper in self.get_modules_wrapper():
            _setattr(self.bound_model, wrapper.name, wrapper.module)
        self.is_wrapped = False

    def compress(self):
        """
        Compress the model with algorithm implemented by subclass.

        The model will be instrumented and user should never edit it after calling this method.
        `self.modules_to_compress` records all the to-be-compressed layers

        Returns
        -------
        torch.nn.Module
            model with specified modules compressed.
        """
        return self.bound_model

    def set_wrappers_attribute(self, name, value):
        """
        To register attributes used in wrapped module's forward method.
        If the type of the value is Torch.tensor, then this value is registered as a buffer in wrapper,
        which will be saved by model.state_dict. Otherwise, this value is just a regular variable in wrapper.

        Parameters
        ----------
        name : str
            name of the variable
        value: any
            value of the variable
        """
        for wrapper in self.get_modules_wrapper():
            if isinstance(value, torch.Tensor):
                wrapper.register_buffer(name, value.clone())
            else:
                setattr(wrapper, name, value)

    def get_modules_to_compress(self):
        """
        To obtain all the to-be-compressed modules.

        Returns
        -------
        list
            a list of the layers, each of which is a tuple (`layer`, `config`),
            `layer` is `LayerInfo`, `config` is a `dict`
        """
        return self.modules_to_compress

    def get_modules_wrapper(self):
        """
        To obtain all the wrapped modules.

        Returns
        -------
        list
            a list of the wrapped modules
        """
        return self.modules_wrapper

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
            # expand config if key `default` is in config['op_types']
            if 'op_types' in config and 'default' in config['op_types']:
                expanded_op_types = []
                for op_type in config['op_types']:
                    if op_type == 'default':
                        expanded_op_types.extend(default_layers.weighted_modules)
                    else:
                        expanded_op_types.append(op_type)
                config['op_types'] = expanded_op_types

            # check if condition is satisified
            if 'op_types' in config and layer.type not in config['op_types']:
                continue
            if 'op_names' in config and layer.name not in config['op_names']:
                continue

            ret = config
        if ret is None or 'exclude' in ret:
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
        pass

    def _wrap_modules(self, layer, config):
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


    def add_activation_collector(self, collector):
        self._fwd_hook_id += 1
        self._fwd_hook_handles[self._fwd_hook_id] = []
        for wrapper in self.get_modules_wrapper():
            handle = wrapper.register_forward_hook(collector)
            self._fwd_hook_handles[self._fwd_hook_id].append(handle)
        return self._fwd_hook_id

    def remove_activation_collector(self, fwd_hook_id):
        if fwd_hook_id not in self._fwd_hook_handles:
            raise ValueError("%s is not a valid collector id" % str(fwd_hook_id))
        for handle in self._fwd_hook_handles[fwd_hook_id]:
            handle.remove()
        del self._fwd_hook_handles[fwd_hook_id]

    def patch_optimizer(self, *tasks):
        def patch_step(old_step):
            def new_step(_, *args, **kwargs):
                # call origin optimizer step method
                output = old_step(*args, **kwargs)
                # calculate mask
                for task in tasks:
                    task()
                return output
            return new_step
        if self.optimizer is not None:
            self.optimizer.step = types.MethodType(patch_step(self.optimizer.step), self.optimizer)

class PrunerModuleWrapper(torch.nn.Module):
    def __init__(self, module, module_name, module_type, config, pruner):
        """
        Wrap an module to enable data parallel, forward method customization and buffer registeration.

        Parameters
        ----------
        module : pytorch module
            the module user wants to compress
        config : dict
            the configurations that users specify for compression
        module_name : str
            the name of the module to compress, wrapper module shares same name
        module_type : str
            the type of the module to compress
        pruner ： Pruner
            the pruner used to calculate mask
        """
        super().__init__()
        # origin layer information
        self.module = module
        self.name = module_name
        self.type = module_type
        # config and pruner
        self.config = config
        self.pruner = pruner

        # register buffer for mask
        self.register_buffer("weight_mask", torch.ones(self.module.weight.shape))
        if hasattr(self.module, 'bias') and self.module.bias is not None:
            self.register_buffer("bias_mask", torch.ones(self.module.bias.shape))
        else:
            self.register_buffer("bias_mask", None)

    def forward(self, *inputs):
        # apply mask to weight, bias
        self.module.weight.data = self.module.weight.data.mul_(self.weight_mask)
        if hasattr(self.module, 'bias') and self.module.bias is not None:
            self.module.bias.data = self.module.bias.data.mul_(self.bias_mask)
        return self.module(*inputs)

class Pruner(Compressor):
    """
    Prune to an exact pruning level specification

    Attributes
    ----------
    mask_dict : dict
        Dictionary for saving masks, `key` should be layer name and
        `value` should be a tensor which has the same shape with layer's weight

    """

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
                    assert hasattr(wrapper, k), "there is no attribute '%s' in wrapper" % k
                    setattr(wrapper, k, masks[k])

    def calc_mask(self, wrapper, **kwargs):
        """
        Pruners should overload this method to provide mask for weight tensors.
        The mask must have the same shape and type comparing to the weight.
        It will be applied with `mul()` operation on the weight.
        This method is effectively hooked to `forward()` method of the model.

        Parameters
        ----------
        wrapper : Module
            calculate mask for `wrapper.module`'s weight
        """
        raise NotImplementedError("Pruners must overload calc_mask()")

    def _wrap_modules(self, layer, config):
        """
        Create a wrapper module to replace the original one.

        Parameters
        ----------
        layer : LayerInfo
            the layer to instrument the mask
        config : dict
            the configuration for generating the mask
        """
        _logger.debug("Module detected to compress : %s.", layer.name)
        wrapper = PrunerModuleWrapper(layer.module, layer.name, layer.type, config, self)
        assert hasattr(layer.module, 'weight'), "module %s does not have 'weight' attribute" % layer.name
        # move newly registered buffers to the same device of weight
        wrapper.to(layer.module.weight.device)
        return wrapper

    def export_model(self, model_path, mask_path=None, onnx_path=None, input_shape=None, device=None):
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
        device : torch.device
            device of the model, used to place the dummy input tensor for exporting onnx file.
            the tensor is placed on cpu if ```device``` is None
        """
        assert model_path is not None, 'model_path must be specified'
        mask_dict = {}
        self._unwrap_model() # used for generating correct state_dict name without wrapper state

        for wrapper in self.get_modules_wrapper():
            weight_mask = wrapper.weight_mask
            bias_mask = wrapper.bias_mask
            if weight_mask is not None:
                mask_sum = weight_mask.sum().item()
                mask_num = weight_mask.numel()
                _logger.debug('Layer: %s  Sparsity: %.4f', wrapper.name, 1 - mask_sum / mask_num)
                wrapper.module.weight.data = wrapper.module.weight.data.mul(weight_mask)
            if bias_mask is not None:
                wrapper.module.bias.data = wrapper.module.bias.data.mul(bias_mask)
            # save mask to dict
            mask_dict[wrapper.name] = {"weight": weight_mask, "bias": bias_mask}

        torch.save(self.bound_model.state_dict(), model_path)
        _logger.info('Model state_dict saved to %s', model_path)
        if mask_path is not None:
            torch.save(mask_dict, mask_path)
            _logger.info('Mask dict saved to %s', mask_path)
        if onnx_path is not None:
            assert input_shape is not None, 'input_shape must be specified to export onnx model'
            # input info needed
            if device is None:
                device = torch.device('cpu')
            input_data = torch.Tensor(*input_shape)
            torch.onnx.export(self.bound_model, input_data.to(device), onnx_path)
            _logger.info('Model in onnx with input shape %s saved to %s', input_data.shape, onnx_path)

        self._wrap_model()

    def load_model_state_dict(self, model_state):
        """
        Load the state dict saved from unwrapped model.

        Parameters:
        -----------
        model_state : dict
            state dict saved from unwrapped model
        """
        if self.is_wrapped:
            self._unwrap_model()
            self.bound_model.load_state_dict(model_state)
            self._wrap_model()
        else:
            self.bound_model.load_state_dict(model_state)

class QuantizerModuleWrapper(torch.nn.Module):
    def __init__(self, module, module_name, module_type, config, quantizer):
        """
        Wrap an module to enable data parallel, forward method customization and buffer registeration.

        Parameters
        ----------
        module : pytorch module
            the module user wants to compress
        config : dict
            the configurations that users specify for compression
        module_name : str
            the name of the module to compress, wrapper module shares same name
        module_type : str
            the type of the module to compress
        quantizer ：quantizer
            the quantizer used to calculate mask
        """
        super().__init__()
        # origin layer information
        self.module = module
        self.name = module_name
        self.type = module_type
        # config and pruner
        self.config = config
        self.quantizer = quantizer

        # register buffer and parameter
        # old_weight is used to store origin weight and weight is used to store quantized weight
        # the reason why weight is buffer instead of parameter is because in pytorch parameter is used as leaf
        # if weight is leaf , then old_weight can not be updated.
        if 'weight' in config['quant_types']:
            if not _check_weight(self.module):
                _logger.warning('Module %s does not have parameter "weight"', self.name)
            else:
                self.module.register_parameter('old_weight', torch.nn.Parameter(self.module.weight))
                delattr(self.module, 'weight')
                self.module.register_buffer('weight', self.module.old_weight)

    def forward(self, *inputs):
        if 'input' in self.config['quant_types']:
            inputs = self.quantizer.quant_grad.apply(
                inputs,
                QuantType.QUANT_INPUT,
                self)

        if 'weight' in self.config['quant_types'] and _check_weight(self.module):
            self.quantizer.quant_grad.apply(
                self.module.old_weight,
                QuantType.QUANT_WEIGHT,
                self)
            result = self.module(*inputs)
        else:
            result = self.module(*inputs)

        if 'output' in self.config['quant_types']:
            result = self.quantizer.quant_grad.apply(
                result,
                QuantType.QUANT_OUTPUT,
                self)
        return result

class Quantizer(Compressor):
    """
    Base quantizer for pytorch quantizer
    """

    def __init__(self, model, config_list, optimizer=None):
        super().__init__(model, config_list, optimizer)
        self.quant_grad = QuantGrad
        if self.optimizer is not None:
            self.patch_optimizer(self.step_with_optimizer)
            for wrapper in self.get_modules_wrapper():
                if 'weight' in wrapper.config['quant_types']:
                    # old_weight is registered to keep track of weight before quantization
                    # and it is trainable, therefore, it should be added to optimizer.
                    self.optimizer.add_param_group({"params": wrapper.module.old_weight})

    def quantize_weight(self, weight, wrapper, **kwargs):
        """
        quantize should overload this method to quantize weight.
        This method is effectively hooked to :meth:`forward` of the model.
        Parameters
        ----------
        weight : Tensor
            weight that needs to be quantized
        wrapper : QuantizerModuleWrapper
            the wrapper for origin module
        """
        raise NotImplementedError('Quantizer must overload quantize_weight()')

    def quantize_output(self, output, wrapper, **kwargs):
        """
        quantize should overload this method to quantize output.
        This method is effectively hooked to :meth:`forward` of the model.
        Parameters
        ----------
        output : Tensor
            output that needs to be quantized
        wrapper : QuantizerModuleWrapper
            the wrapper for origin module
        """
        raise NotImplementedError('Quantizer must overload quantize_output()')

    def quantize_input(self, *inputs, wrapper, **kwargs):
        """
        quantize should overload this method to quantize input.
        This method is effectively hooked to :meth:`forward` of the model.
        Parameters
        ----------
        inputs : Tensor
            inputs that needs to be quantized
        wrapper : QuantizerModuleWrapper
            the wrapper for origin module
        """
        raise NotImplementedError('Quantizer must overload quantize_input()')


    def _wrap_modules(self, layer, config):
        """
        Create a wrapper forward function to replace the original one.
        Parameters
        ----------
        layer : LayerInfo
            the layer to instrument the mask
        config : dict
            the configuration for quantization
        """
        assert 'quant_types' in config, 'must provide quant_types in config'
        assert isinstance(config['quant_types'], list), 'quant_types must be list type'
        assert 'quant_bits' in config, 'must provide quant_bits in config'
        assert isinstance(config['quant_bits'], int) or isinstance(config['quant_bits'], dict), 'quant_bits must be dict type or int type'

        if isinstance(config['quant_bits'], dict):
            for quant_type in config['quant_types']:
                assert quant_type in config['quant_bits'], 'bits length for %s must be specified in quant_bits dict' % quant_type

        return QuantizerModuleWrapper(layer.module, layer.name, layer.type, config, self)

    def step_with_optimizer(self):
        pass

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
    def forward(ctx, tensor, quant_type, wrapper, **kwargs):
        ctx.save_for_backward(tensor, torch.Tensor([quant_type]))
        if quant_type == QuantType.QUANT_INPUT:
            return wrapper.quantizer.quantize_input(tensor, wrapper, **kwargs)
        elif quant_type == QuantType.QUANT_WEIGHT:
            return wrapper.quantizer.quantize_weight(wrapper, **kwargs)
        elif quant_type == QuantType.QUANT_OUTPUT:
            return wrapper.quantizer.quantize_output(tensor, wrapper, **kwargs)
        else:
            raise ValueError("unrecognized QuantType.")

    @classmethod
    def backward(cls, ctx, grad_output):
        tensor, quant_type = ctx.saved_variables
        output = cls.quant_backward(tensor, grad_output, quant_type)
        return output, None, None, None

def _check_weight(module):
    try:
        return isinstance(module.weight.data, torch.Tensor)
    except AttributeError:
        return False
