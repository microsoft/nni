# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module

from .compressor import Compressor, LayerInfo

_logger = logging.getLogger(__name__)

__all__ = ['Pruner']


class PrunerModuleWrapper(Module):
    def __init__(self, module: Module, module_name: str, config: Dict, pruner: Compressor):
        """
        Wrap an module to enable data parallel, forward method customization and buffer registeration.

        Parameters
        ----------
        module
            The module user wants to compress.
        config
            The configurations that users specify for compression.
        module_name
            The name of the module to compress, wrapper module shares same name.
        pruner
            The pruner used to calculate mask.
        """
        super().__init__()
        # origin layer information
        self.module = module
        self.name = module_name
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
    The abstract class for pruning algorithm. Inherit this class and implement the `_reset_tools` to customize a pruner.
    """

    def reset(self, model: Optional[Module] = None, config_list: Optional[List[Dict]] = None):
        super().reset(model=model, config_list=config_list)

    def _wrap_modules(self, layer: LayerInfo, config: Dict):
        """
        Create a wrapper module to replace the original one.

        Parameters
        ----------
        layer
            The layer to instrument the mask.
        config
            The configuration for generating the mask.
        """
        _logger.debug("Module detected to compress : %s.", layer.name)
        wrapper = PrunerModuleWrapper(layer.module, layer.name, config, self)
        assert hasattr(layer.module, 'weight'), "module %s does not have 'weight' attribute" % layer.name
        # move newly registered buffers to the same device of weight
        wrapper.to(layer.module.weight.device)
        return wrapper

    def load_masks(self, masks: Dict[str, Dict[str, Tensor]]):
        """
        Load an exist masks on the wrapper. You can train the model with an exist masks after load the masks.

        Parameters
        ----------
        masks
            The masks dict with format {'op_name': {'weight_mask': mask, 'bias_mask': mask}}.
        """
        wrappers = self.get_modules_wrapper()
        for name, layer_mask in masks.items():
            assert name in wrappers, '{} is not in wrappers of this pruner, can not apply the mask.'.format(name)
            for mask_type, mask in layer_mask.items():
                assert hasattr(wrappers[name], mask_type), 'there is no attribute {} in wrapper'.format(mask_type)
                setattr(wrappers[name], mask_type, mask)

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
        for _, wrapper in self.get_modules_wrapper().items():
            weight_mask = wrapper.weight_mask
            mask_size = weight_mask.size()
            if len(mask_size) == 1:
                index = torch.nonzero(weight_mask.abs() != 0, as_tuple=False).tolist()
            else:
                sum_idx = list(range(len(mask_size)))
                sum_idx.remove(dim)
                index = torch.nonzero(weight_mask.abs().sum(sum_idx) != 0, as_tuple=False).tolist()
            _logger.info(f'simulated prune {wrapper.name} remain/total: {len(index)}/{weight_mask.size(dim)}')

    def export_model(self, model_path, mask_path=None, onnx_path=None, input_shape=None, device=None):
        """
        Export pruned model weights, masks and onnx model(optional)

        Parameters
        ----------
        model_path
            Path to save pruned model state_dict.
        mask_path
            (optional) path to save mask dict.
        onnx_path
            (optional) path to save onnx model.
        input_shape
            Input shape to onnx model.
        device
            Device of the model, used to place the dummy input tensor for exporting onnx file.
            The tensor is placed on cpu if ```device``` is None.
        """
        assert model_path is not None, 'model_path must be specified'
        mask_dict = {}
        self._unwrap_model()  # used for generating correct state_dict name without wrapper state

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
            mask_dict[name] = {"weight_mask": weight_mask, "bias_mask": bias_mask}

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
