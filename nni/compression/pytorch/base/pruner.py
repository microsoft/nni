# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Dict, List, Optional, Tuple, OrderedDict

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter

from .compressor import Compressor, LayerInfo, _setattr

_logger = logging.getLogger(__name__)

__all__ = ['Pruner']


class PrunerModuleWrapper(Module):
    """
    Wrap a module to enable data parallel, forward method customization and buffer registeration.

    Parameters
    ----------
    module
        The module user wants to compress.
    config
        The configurations that users specify for compression.
    module_name
        The name of the module to compress, wrapper module shares same name.
    """

    def __init__(self, module: Module, module_name: str, config: Dict):
        super().__init__()
        # origin layer information
        self.module = module
        self.name = module_name
        # config information
        self.config = config

        pruning_target_names = ['weight', 'bias']
        for pruning_target_name in pruning_target_names:
            pruning_target_mask_name = '{}_mask'.format(pruning_target_name)
            pruning_target = getattr(self.module, pruning_target_name, None)
            if hasattr(self.module, pruning_target_name) and pruning_target is not None:
                setattr(self, pruning_target_name, Parameter(torch.empty_like(pruning_target)))
                self.register_buffer(pruning_target_mask_name, torch.ones_like(pruning_target))
            else:
                self.register_buffer(pruning_target_mask_name, None)

    def _weight2buffer(self):
        """
        When using this wrapper to inference, call `_weight2buffer()` to make original weight untrainable.
        The best place to call this function is in `Pruner._wrap_model()`.
        """
        self.weight.data = self.module.weight.data
        delattr(self.module, 'weight')
        self.module.register_buffer('weight', self.weight.data)
        if hasattr(self.module, 'bias') and self.module.bias is not None:
            self.bias.data = self.module.bias.data
            delattr(self.module, 'bias')
            self.module.register_buffer('bias', self.bias.data)

    def _weight2parameter(self):
        """
        When don't need to record score or need to export the model, call `_weight2parameter()` to make the original weight trainable.
        The best place to call this function is in `Pruner._unwrap_model()`.
        """
        delattr(self.module, 'weight')
        self.module.weight = Parameter(torch.empty_like(self.weight))
        self.module.weight.data = torch.mul(self.weight, self.weight_mask)
        if hasattr(self.module, 'bias') and self.module.bias is not None:
            delattr(self.module, 'bias')
            self.module.bias = Parameter(torch.empty_like(self.bias))
            self.module.bias.data = torch.mul(self.bias, self.bias_mask)

    def forward(self, *inputs):
        # apply mask to weight, bias
        self.module.weight = torch.mul(self.weight, self.weight_mask)
        if hasattr(self.module, 'bias') and self.module.bias is not None:
            self.module.bias = torch.mul(self.bias, self.bias_mask)
        return self.module(*inputs)


class Pruner(Compressor):
    """
    The abstract class for pruning algorithm. Inherit this class and implement the `_reset_tools` to customize a pruner.
    """

    def reset(self, model: Optional[Module] = None, config_list: Optional[List[Dict]] = None):
        super().reset(model=model, config_list=config_list)

    def get_modules_wrapper(self) -> OrderedDict[str, PrunerModuleWrapper]:
        """
        Returns
        -------
        OrderedDict[str, PrunerModuleWrapper]
            An ordered dict, key is the name of the module, value is the wrapper of the module.
        """
        assert self.modules_wrapper is not None, 'Bound model has not be wrapped.'
        return self.modules_wrapper

    def _wrap_modules(self, layer: LayerInfo, config: Dict) -> PrunerModuleWrapper:
        """
        Create a wrapper module to replace the original one.

        Parameters
        ----------
        layer
            The layer to instrument the mask.
        config
            The configuration for generating the mask.

        Returns
        -------
        PrunerModuleWrapper
            The wrapper of the module in layerinfo.
        """
        _logger.debug("Module detected to compress : %s.", layer.name)
        wrapper = PrunerModuleWrapper(layer.module, layer.name, config)
        assert hasattr(layer.module, 'weight'), "module %s does not have 'weight' attribute" % layer.name
        # move newly registered buffers to the same device of weight
        wrapper.to(layer.module.weight.device)
        return wrapper

    # The following `_wrap_model`, `_unwrap_model`, `get_origin2wrapped_parameter_name_map` can merge to `Compressor`,
    # if quantizer use the similar structure wrapper.
    def _wrap_model(self):
        """
        Wrap all modules that needed to be compressed.
        Different from the parent function, call `wrapper._weight2buffer()` after replace the origin module to wrapper.
        """
        err_msg = 'No model bounded in this compressor, please use Compressor.reset(model, config_list) to set it.'
        assert self.bound_model is not None, err_msg

        if not self.is_wrapped:
            for _, wrapper in reversed(list(self.get_modules_wrapper().items())):
                _setattr(self.bound_model, wrapper.name, wrapper)
                wrapper._weight2buffer()
            self.is_wrapped = True

    def _unwrap_model(self):
        """
        Unwrap all modules that needed to be compressed.
        Different from the parent function, call `wrapper._weight2parameter()` after replace the wrapper to origin module.
        """
        err_msg = 'No model bounded in this compressor, please use Compressor.reset(model, config_list) to set it.'
        assert self.bound_model is not None, err_msg

        if self.is_wrapped:
            for wrapper in self.get_modules_wrapper().values():
                _setattr(self.bound_model, wrapper.name, wrapper.module)
                wrapper._weight2parameter()
            self.is_wrapped = False

    def get_origin2wrapped_parameter_name_map(self) -> Dict[str, str]:
        """
        Get the name mapping of parameters from original model to wrapped model.

        Returns
        -------
        Dict[str, str]
            Return a dict `{original_model_parameter_name: wrapped_model_parameter_name}`
        """
        if self.is_wrapped:
            wrapped_param_names = {id(param): name for name, param in self.bound_model.named_parameters()}
            self._unwrap_model()
            parameter_name_map = {}
            for name, param in self.bound_model.named_parameters():
                # If the parameter name in under wrapped module is `xxx.weight` or `xxx.bias`,
                # the name will not change after wrap.
                # If the parameter name in under wrapped module is others,
                # the name `xxx.param` will change to `xxx.module.param` after wrap.
                parameter_name_map[name] = wrapped_param_names[id(param)] if id(param) in wrapped_param_names else name
            self._wrap_model()
            return parameter_name_map
        else:
            raise Exception('When only the model is wrapped can get the parameter_name_map.')

    def load_masks(self, masks: Dict[str, Dict[str, Tensor]]):
        """
        Load an exist masks on the wrapper. You can train the model with an exist masks after load the masks.

        Parameters
        ----------
        masks
            The masks dict with format {'op_name': {'weight': mask, 'bias': mask}}.
        """
        wrappers = self.get_modules_wrapper()
        for module_name, target_masks in masks.items():
            assert module_name in wrappers, '{} is not in wrappers of this pruner, can not apply the mask.'.format(module_name)
            for target_name, target_mask in target_masks.items():
                assert hasattr(wrappers[module_name], f'{target_name}_mask'), f'There is no attribute {target_name}_mask in wrapper.'
                target: Tensor = getattr(self.get_modules_wrapper()[module_name], target_name)
                setattr(wrappers[module_name], f'{target_name}_mask', target_mask.to(target.device))

    def compress(self) -> Tuple[Module, Dict[str, Dict[str, Tensor]]]:
        """
        Returns
        -------
        Tuple[Module, Dict]
            Return the wrapped model and mask.
        """
        return self.bound_model, {}

    # NOTE: need refactor dim with supporting list
    def show_pruned_weights(self, dim: int = 0):
        """
        Log the simulated prune sparsity.

        Parameters
        ----------
        dim
            The pruned dim.
        """
        for wrapper in self.get_modules_wrapper().values():
            weight_mask = wrapper.weight_mask
            mask_size = weight_mask.size()
            if len(mask_size) == 1:
                index = torch.nonzero(weight_mask.abs() != 0, as_tuple=False).tolist()
            else:
                sum_idx = list(range(len(mask_size)))
                sum_idx.remove(dim)
                index = torch.nonzero(weight_mask.abs().sum(sum_idx) != 0, as_tuple=False).tolist()
            _logger.info(f'simulated prune {wrapper.name} remain/total: {len(index)}/{weight_mask.size(dim)}')

    def export_model(self, model_path: str, mask_path: Optional[str] = None):
        """
        Export pruned model weights, masks and onnx model(optional)

        Parameters
        ----------
        model_path
            Path to save pruned model state_dict. The weight and bias have already multiplied the masks.
        mask_path
            Path to save mask dict.
        """
        assert self.bound_model is not None, 'The bound model reference has been cleared.'
        assert model_path is not None, 'model_path must be specified.'
        mask_dict = {}
        self._unwrap_model()

        for name, wrapper in self.get_modules_wrapper().items():
            weight_mask = wrapper.weight_mask
            bias_mask = wrapper.bias_mask
            if weight_mask is not None:
                mask_sum = weight_mask.sum().item()
                mask_num = weight_mask.numel()
                _logger.debug('Layer: %s  Sparsity: %.4f', name, 1 - mask_sum / mask_num)
                wrapper.module.weight.data = wrapper.module.weight.data.mul(weight_mask)
            if bias_mask is not None:
                wrapper.module.bias.data = wrapper.module.bias.data.mul(bias_mask)
            # save mask to dict
            mask_dict[name] = {"weight": weight_mask, "bias": bias_mask}

        torch.save(self.bound_model.state_dict(), model_path)
        _logger.info('Model state_dict saved to %s', model_path)

        if mask_path is not None:
            torch.save(mask_dict, mask_path)
            _logger.info('Mask dict saved to %s', mask_path)

        self._wrap_model()
