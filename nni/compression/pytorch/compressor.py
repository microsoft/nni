# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import types
import logging
import torch
from nni.common.graph_utils import build_module_graph
from nni.compression.pytorch.quantization.literal import QuantType, BN_FOLD_OP, BN_FOLD_TAG
from nni.compression.pytorch.quantization.observers import RecordingObserver
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

    def patch_optimizer_before(self, *tasks):
        def patch_step(old_step):
            def new_step(_, *args, **kwargs):
                for task in tasks:
                    task()
                # call origin optimizer step method
                output = old_step(*args, **kwargs)
                return output
            return new_step
        if self.optimizer is not None:
            self.optimizer.step = types.MethodType(patch_step(self.optimizer.step), self.optimizer)

class PrunerModuleWrapper(torch.nn.Module):
    def __init__(self, module, module_name, module_type, config, pruner):
        """
        Wrap a module to enable data parallel, forward method customization and buffer registeration.

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

    def export_model(self, model_path, mask_path=None, onnx_path=None, input_shape=None, device=None,
                     dummy_input=None, opset_version=None):
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
            input shape to onnx model, used for creating a dummy input tensor for torch.onnx.export
            if the input has a complex structure (e.g., a tuple), please directly create the input and
            pass it to dummy_input instead
            note: this argument is deprecated and will be removed; please use dummy_input instead
        device : torch.device
            device of the model, where to place the dummy input tensor for exporting onnx file;
            the tensor is placed on cpu if ```device``` is None
            only useful when both onnx_path and input_shape are passed
            note: this argument is deprecated and will be removed; please use dummy_input instead
        dummy_input: torch.Tensor or tuple
            dummy input to the onnx model; used when input_shape is not enough to specify dummy input
            user should ensure that the dummy_input is on the same device as the model
        opset_version: int
            opset_version parameter for torch.onnx.export; only useful when onnx_path is not None
            if not passed, torch.onnx.export will use its default opset_version
        """
        assert model_path is not None, 'model_path must be specified'
        mask_dict = {}
        self._unwrap_model()  # used for generating correct state_dict name without wrapper state

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
            assert input_shape is not None or dummy_input is not None,\
                'input_shape or dummy_input must be specified to export onnx model'
            # create dummy_input using input_shape if input_shape is not passed
            if dummy_input is None:
                _logger.warning("""The argument input_shape and device will be removed in the future.
                                   Please create a dummy input and pass it to dummy_input instead.""")
                if device is None:
                    device = torch.device('cpu')
                input_data = torch.Tensor(*input_shape).to(device)
            else:
                input_data = dummy_input
            if opset_version is not None:
                torch.onnx.export(self.bound_model, input_data, onnx_path, opset_version=opset_version)
            else:
                torch.onnx.export(self.bound_model, input_data, onnx_path)
            if dummy_input is None:
                _logger.info('Model in onnx with input shape %s saved to %s', input_data.shape, onnx_path)
            else:
                _logger.info('Model in onnx saved to %s', onnx_path)

        self._wrap_model()

    def load_model_state_dict(self, model_state):
        """
        Load the state dict saved from unwrapped model.

        Parameters
        ----------
        model_state : dict
            state dict saved from unwrapped model
        """
        if self.is_wrapped:
            self._unwrap_model()
            self.bound_model.load_state_dict(model_state)
            self._wrap_model()
        else:
            self.bound_model.load_state_dict(model_state)

    def get_pruned_weights(self, dim=0):
        """
        Log the simulated prune sparsity.

        Parameters
        ----------
        dim : int
            the pruned dim.
        """
        for _, wrapper in enumerate(self.get_modules_wrapper()):
            weight_mask = wrapper.weight_mask
            mask_size = weight_mask.size()
            if len(mask_size) == 1:
                index = torch.nonzero(weight_mask.abs() != 0).tolist()
            else:
                sum_idx = list(range(len(mask_size)))
                sum_idx.remove(dim)
                index = torch.nonzero(weight_mask.abs().sum(sum_idx) != 0).tolist()
            _logger.info(f'simulated prune {wrapper.name} remain/total: {len(index)}/{weight_mask.size(dim)}')


class QuantizerModuleWrapper(torch.nn.Module):
    def __init__(self, module, module_name, module_type, config, quantizer, bn_module=None):
        """
        Wrap a module to enable data parallel, forward method customization and buffer registeration.

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
        bn_module : torch.nn.Module
            batch norm layer corresponding to current module, used for simulating batch normalization folding
        """
        super().__init__()
        # origin layer information
        self.module = module
        self.name = module_name
        self.type = module_type
        # config and pruner
        self.config = config
        self.quantizer = quantizer
        self.bn_module = bn_module

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
                self.module.register_buffer('weight', self.module.old_weight.data)

                # for batch normalization folding
                if self.bn_module is not None:
                    if _check_bias(self.module):
                        self.module.register_parameter('old_bias', torch.nn.Parameter(self.module.bias))
                        init_tensor = self.module.old_bias.data
                    else:
                        init_tensor = torch.zeros_like(self.bn_module.weight)
                    delattr(self.module, 'bias')
                    self.module.register_buffer('bias', init_tensor)
                    setattr(module, BN_FOLD_TAG, True)

    def forward(self, *inputs):
        if 'input' in self.config['quant_types']:
            assert len(inputs) == 1, "Quantization of input only supports ops with single input."
            new_inp = self.quantizer.quant_grad(
                inputs[0],
                QuantType.INPUT,
                self)
            inputs = (new_inp,)

        if 'weight' in self.config['quant_types'] and _check_weight(self.module):
            if self.bn_module is not None:
                # simulate batch normalization folding
                new_weight, new_bias = self.quantizer.fold_bn(*inputs, wrapper=self)
                self.module.bias = new_bias
                self.module.weight = new_weight
            else:
                new_weight = self.module.old_weight
                self.module.weight = new_weight.data

            self.quantizer.quant_grad(
                new_weight,
                QuantType.WEIGHT,
                self, inputs[0])

        result = self.module(*inputs)

        if 'output' in self.config['quant_types']:
            result = self.quantizer.quant_grad(
                result,
                QuantType.OUTPUT,
                self)
        return result


class QuantizerIdentityWrapper(torch.nn.Module):
    def __init__(self, module, module_name):
        """
        Used to wrap modules that should be treated as torch.Identity

        Parameters
        ----------
        module : pytorch module
            the module to be wrapped
        module_name : str
            the name of the module to wrapped, wrapper module shares same name
        """
        super().__init__()
        self.module = module
        self.module_name = module_name

    def forward(self, x):
        return x


class Quantizer(Compressor):
    """
    Base quantizer for pytorch quantizer
    """

    def __init__(self, model, config_list, optimizer=None, dummy_input=None):
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        model_copied = copy.deepcopy(model)
        self.identity_wrappers = []
        self.conv_bn_patterns = {}
        self.find_conv_bn_patterns(model, dummy_input)
        super().__init__(model, config_list, optimizer)
        self.all_shapes = {}
        self.record_shape(model_copied, dummy_input)
        self.quant_grad = QuantGrad.apply
        if self.optimizer is not None:
            self.patch_optimizer(self.step_with_optimizer)
            for wrapper in self.get_modules_wrapper():
                if 'weight' in wrapper.config['quant_types']:
                    # old_weight is registered to keep track of weight before quantization
                    # and it is trainable, therefore, it should be added to optimizer.
                    self.optimizer.add_param_group({"params": wrapper.module.old_weight})
                # This is for conv with bias + bn. Although this situation is relatively rare,
                # we still need to deal with the old_bias when it occurs
                if hasattr(wrapper.module, "old_bias"):
                    self.optimizer.add_param_group({"params": getattr(wrapper.module, "old_bias")})

    def quantize_weight(self, wrapper, **kwargs):
        """
        quantize should overload this method to quantize weight.
        This method is effectively hooked to :meth:`forward` of the model.
        Parameters
        ----------
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

    def quantize_input(self, inputs, wrapper, **kwargs):
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

    def fold_bn(self, *inputs, wrapper):
        """
        Simulate batch normalization folding in the training graph. Folded weight and bias are
        returned for the following operations.

        Parameters
        ----------
        inputs : tuple of torch.Tensor
            inputs for the module
        wrapper : QuantizerModuleWrapper
            the wrapper for origin module

        Returns
        -------
        Tuple of torch.Tensor
        """
        module = wrapper.module
        bn_module = wrapper.bn_module
        with torch.no_grad():
            output = module(*inputs)
            _ = bn_module(output)
        running_mean = bn_module.running_mean
        running_var = torch.sqrt(bn_module.running_var + bn_module.eps)
        bn_weight = bn_module.weight
        bn_bias = bn_module.bias
        dimensions = len(module.weight.shape)
        shape = [-1] + [1] * (dimensions - 1)
        new_weight = module.old_weight * bn_weight.reshape(shape) / running_var.reshape(shape)
        if hasattr(module, 'old_bias'):
            new_bias = bn_bias + (module.old_bias - running_mean) / running_var * bn_weight
        else:
            new_bias = bn_bias - running_mean / running_var * bn_weight
        return new_weight, new_bias

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

        # bound bn module to corresponding conv module
        bn_module = None
        if layer.name in self.conv_bn_patterns:
            bn_module_name = self.conv_bn_patterns[layer.name]
            for name, module in self.bound_model.named_modules():
                if name == bn_module_name:
                    bn_module = module
                    break
            assert bn_module is not None, "BN module corresponding to layer {} is not found".format(layer.name)
            self.identity_wrappers.append(QuantizerIdentityWrapper(bn_module, bn_module_name))
        return QuantizerModuleWrapper(layer.module, layer.name, layer.type, config, self, bn_module)

    def _wrap_model(self):
        """
        wrap all modules that needed to be compressed

        """
        # wrap folded bn in order to bypass its forward process
        for wrapper in reversed(self.identity_wrappers):
            _setattr(self.bound_model, wrapper.module_name, wrapper)
        super()._wrap_model()

    def _unwrap_model(self):
        """
        unwrap all modules that needed to be compressed

        """
        for wrapper in self.identity_wrappers:
            _setattr(self.bound_model, wrapper.module_name, wrapper.module)
        super()._unwrap_model()

    def export_model_save(self, model, model_path, calibration_config=None, calibration_path=None, onnx_path=None,
                          input_shape=None, device=None):
        """
        This method helps save pytorch model, calibration config, onnx model in quantizer.

        Parameters
        ----------
        model : pytorch model
            pytorch model to be saved
        model_path : str
            path to save pytorch
        calibration_config: dict
            (optional) config of calibration parameters
        calibration_path : str
            (optional) path to save quantize parameters after calibration
        onnx_path : str
            (optional) path to save onnx model
        input_shape : list or tuple
            input shape to onnx model
        device : torch.device
            device of the model, used to place the dummy input tensor for exporting onnx file.
            the tensor is placed on cpu if ```device``` is None
        """
        torch.save(model.state_dict(), model_path)
        _logger.info('Model state_dict saved to %s', model_path)
        if calibration_path is not None:
            torch.save(calibration_config, calibration_path)
            _logger.info('Mask dict saved to %s', calibration_path)
        if onnx_path is not None:
            assert input_shape is not None, 'input_shape must be specified to export onnx model'
            # input info needed
            if device is None:
                device = torch.device('cpu')
            input_data = torch.Tensor(*input_shape)
            torch.onnx.export(self.bound_model, input_data.to(device), onnx_path)
            _logger.info('Model in onnx with input shape %s saved to %s', input_data.shape, onnx_path)

    def export_model(self, model_path, calibration_path=None, onnx_path=None, input_shape=None, device=None):
        """
        Export quantized model weights and calibration parameters

        Parameters
        ----------
        model_path : str
            path to save quantized model weight
        calibration_path : str
            (optional) path to save quantize parameters after calibration
        onnx_path : str
            (optional) path to save onnx model
        input_shape : list or tuple
            input shape to onnx model
        device : torch.device
            device of the model, used to place the dummy input tensor for exporting onnx file.
            the tensor is placed on cpu if ```device``` is None

        Returns
        -------
        Dict
        """
        raise NotImplementedError('Quantizer must overload export_model()')

    def load_calibration_config(self, calibration_config):
        """
        This function aims to help quantizer set quantization parameters by
        loading from a calibration_config which is exported by other quantizer
        or itself. The main usage of this function is helping quantize aware training
        quantizer set appropriate initial parameters so that the training process will
        be much more flexible and converges quickly. What's more, it can also enable
        quantizer resume quantization model by loading parameters from config.

        Parameters
        ----------
        calibration_config : dict
            dict which saves quantization parameters, quantizer can export itself
            calibration config.
            eg, calibration_config = quantizer.export_model(model_path, calibration_path)
        """
        raise NotImplementedError('Quantizer must overload export_model()')

    def find_conv_bn_patterns(self, model, dummy_input):
        """
        Find all Conv-BN patterns, used for batch normalization folding

        Parameters
        ----------
        model : torch.nn.Module
            model to be analyzed.
        dummy_input : tupel of torch.tensor
            inputs to the model, used for generating the torchscript
        """
        if dummy_input is None:
            _logger.debug("Model inputs are not given, batch normalization folding is disabled")
            return

        graph = build_module_graph(model, dummy_input)
        for node_group in graph.nodes_py.nodes_op:
            if node_group.op_type in BN_FOLD_OP:
                successors = graph.find_successors(node_group.unique_name)
                successors = [graph.name_to_node[x] for x in successors]
                for successor in successors:
                    if successor.op_type == 'BatchNorm2d':
                        self.conv_bn_patterns[node_group.name] = successor.name

    def record_shape(self, model, dummy_input):
        """
        Record input/output's shapes of each module to be quantized

        Parameters
        ----------
        model : torch.nn.Module
            model to be recorded.
        dummy_input : tupel of torch.tensor
            inputs to the model.
        """
        def _pre_forward_hook(self, inp):
            # Only record the first tensor of the input
            return self.pre_forward(inp[0])

        def _post_forward_hook(self, _, out):
            return self.post_forward(out)

        if dummy_input is None:
            return

        all_handles = []
        all_observers = {}
        modules_to_compress = self.get_modules_to_compress()
        compress_names = [layer_info[0].name for layer_info in modules_to_compress]
        for name, module in model.named_modules():
            if name in compress_names:
                all_observers[name] = {}
                all_observers[name]['input_hook'] = RecordingObserver()
                all_observers[name]['output_hook'] = RecordingObserver()
                module.add_module('pre_forward', all_observers[name]['input_hook'])
                module.add_module('post_forward', all_observers[name]['output_hook'])
                all_handles.append(module.register_forward_pre_hook(_pre_forward_hook))
                all_handles.append(module.register_forward_hook(_post_forward_hook))
        model(dummy_input)
        for name, hooks in all_observers.items():
            # only support single input
            input_val = hooks['input_hook'].tensor_val
            input_shape = input_val[0].shape if input_val else None
            output_val = hooks['output_hook'].tensor_val
            output_shape = output_val[0].shape if output_val else None
            shapes = [input_shape, output_shape]
            self.all_shapes[name] = shapes
        return

    def step_with_optimizer(self):
        pass


class QuantGrad(torch.autograd.Function):
    """
    Base class for overriding backward function of quantization operation.
    """
    @classmethod
    def _quantize(cls, x, scale, zero_point):
        """
        Reference function for quantizing x -- non-clamped.
        Parameters
        ----------
        x : Tensor
            tensor to be quantized
        scale : Tensor
            scale for quantizing x
        zero_point : Tensor
            zero_point for quantizing x
        Returns
        -------
        tensor
            quantized x without clamped
        """
        return ((x / scale) + zero_point).round()

    @classmethod
    def get_bits_length(cls, config, quant_type):
        """
        Get bits for quantize config
        Parameters
        ----------
        config : Dict
            the configuration for quantization
        quant_type : str
            quant type
        Returns
        -------
        int
            n-bits for quantization configuration
        """
        if isinstance(config["quant_bits"], int):
            return config["quant_bits"]
        else:
            return config["quant_bits"].get(quant_type)

    @staticmethod
    def quant_backward(tensor, grad_output, quant_type, scale, zero_point, qmin, qmax):
        """
        This method should be overrided by subclass to provide customized backward function,
        default implementation is Straight-Through Estimator
        Parameters
        ----------
        tensor : Tensor
            input of quantization operation
        grad_output : Tensor
            gradient of the output of quantization operation
        scale : Tensor
            the type of quantization, it can be `QuantType.INPUT`, `QuantType.WEIGHT`,
            `QuantType.OUTPUT`, you can define different behavior for different types.
        zero_point : Tensor
            zero_point for quantizing tensor
        qmin : Tensor
            quant_min for quantizing tensor
        qmax : Tensor
            quant_max for quantizng tensor
        Returns
        -------
        tensor
            gradient of the input of quantization operation
        """
        return grad_output

    @staticmethod
    def forward(ctx, tensor, quant_type, wrapper, input_tensor=None, **kwargs):
        output = quantize_helper(tensor, quant_type, wrapper, input_tensor, **kwargs)

        if hasattr(wrapper.module, "layer_quant_setting"):
            layer_quant_setting = wrapper.module.layer_quant_setting
            qmin, qmax = getattr(layer_quant_setting, quant_type).get_qmin_qmax()
        else:
            # todo: when dtype/scheme customization is ready for all quantizers, remove this
            bits = QuantGrad.get_bits_length(wrapper.config, quant_type)
            qmin, qmax = 0, (1 << bits) - 1

        scale_name, zero_point_name = quant_type.type_to_scale_zero_point_name()
        if hasattr(wrapper.module, scale_name) and hasattr(wrapper.module, zero_point_name):
            scale = getattr(wrapper.module, scale_name)
            zero_point = getattr(wrapper.module, zero_point_name)
        # todo: remove this when other quantizers use different scale & zero point for input/weight/output
        elif hasattr(wrapper.module, 'scale') and hasattr(wrapper.module, 'zero_point'):
            scale = wrapper.module.scale
            zero_point = wrapper.module.zero_point
        else:
            scale, zero_point = None, None
        # Only tensors have gradients flowing back needs to be saved by save_for_backward.
        # Others should directly assign to ctx.
        ctx.save_for_backward(tensor)
        ctx.quant_type = quant_type
        ctx.qmin, ctx.qmax = qmin, qmax
        ctx.scale = scale
        ctx.zero_point = zero_point
        return output

    @classmethod
    def backward(cls, ctx, grad_output):
        tensor = ctx.saved_variables[0]
        scale, zero_point = ctx.scale, ctx.zero_point
        quant_type = ctx.quant_type
        qmin, qmax = ctx.qmin, ctx.qmax
        output = cls.quant_backward(tensor, grad_output, quant_type, scale, zero_point, qmin, qmax)
        return output, None, None, None

def _check_weight(module):
    try:
        return isinstance(module.weight.data, torch.Tensor)
    except AttributeError:
        return False

def _check_bias(module):
    try:
        return isinstance(module.bias.data, torch.Tensor)
    except AttributeError:
        return False

def quantize_helper(tensor, quant_type, wrapper, input_tensor=None, **kwargs):
    if quant_type == QuantType.INPUT:
        output = wrapper.quantizer.quantize_input(tensor, wrapper=wrapper, **kwargs)
    elif quant_type == QuantType.WEIGHT:
        output = wrapper.quantizer.quantize_weight(wrapper, input_tensor=input_tensor, **kwargs)
    elif quant_type == QuantType.OUTPUT:
        output = wrapper.quantizer.quantize_output(tensor, wrapper, **kwargs)
    else:
        raise ValueError("unrecognized QuantType.")

    return output

class QuantForward(torch.nn.Module):
    """
    Base class for executing quantization operations. This is for quantization algorithms
    that do not need to customize gradient.
    """

    def forward(self, tensor, quant_type, wrapper, input_tensor=None, **kwargs):
        return quantize_helper(tensor, quant_type, wrapper, input_tensor, **kwargs)
