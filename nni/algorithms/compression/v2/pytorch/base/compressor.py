# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import collections
import logging
from typing import List, Dict, Optional, Tuple, Any

import torch
from torch.nn import Module

from nni.common.graph_utils import TorchModuleGraph
from nni.algorithms.compression.v2.pytorch.utils.pruning import get_module_by_name, weighted_modules

_logger = logging.getLogger(__name__)

__all__ = ['LayerInfo', 'Compressor']


class LayerInfo:
    def __init__(self, name: str, module: Module):
        self.module = module
        self.name = name
        self.type = type(module).__name__


def _setattr(model: Module, name: str, module: Module):
    parent_module, _ = get_module_by_name(model, name)
    if parent_module is not None:
        name_list = name.split(".")
        setattr(parent_module, name_list[-1], module)
    else:
        raise '{} not exist.'.format(name)


class Compressor:
    """
    The abstract base pytorch compressor.
    """

    def __init__(self, model: Optional[Module], config_list: Optional[List[Dict]]):
        """
        Parameters
        ----------
        model
            The model under compressed.
        config_list
            The config list used by compressor, usually specifies the 'op_types' or 'op_names' that want to compress.
        """
        self.is_wrapped = False
        if model is not None:
            self.reset(model=model, config_list=config_list)
        else:
            _logger.warning('This compressor is not set model and config_list, waiting for reset() or pass this to scheduler.')

    def reset(self, model: Module, config_list: List[Dict]):
        """
        Reset the compressor with model and config_list.

        Parameters
        ----------
        model
            The model under compressed.
        config_list
            The config list used by compressor, usually specifies the 'op_types' or 'op_names' that want to compress.
        """
        assert isinstance(model, Module), 'Only support compressing pytorch Module, but the type of model is {}.'.format(type(model))
        self.bound_model = model
        self.config_list = config_list
        self.validate_config(model=model, config_list=config_list)

        self._unwrap_model()

        self._modules_to_compress = None
        self.modules_wrapper = collections.OrderedDict()
        for layer, config in self._detect_modules_to_compress():
            wrapper = self._wrap_modules(layer, config)
            self.modules_wrapper[layer.name] = wrapper

        self._wrap_model()

    def clear_model_references(self):
        """
        Clear all references to the model in this compressor. Just to free up memory.
        Need reset first before the next time call compressor function.
        """
        self._unwrap_model()
        self.bound_model = None
        self.config_list = None
        self.modules_wrapper = None
        self._modules_to_compress = None

    def _detect_modules_to_compress(self) -> List[Tuple[LayerInfo, Dict]]:
        """
        Detect all modules should be compressed, and save the result in `self._modules_to_compress`.
        The model will be instrumented and user should never edit it after calling this method.
        """
        if self._modules_to_compress is None:
            self._modules_to_compress = []
            for name, module in self.bound_model.named_modules():
                if module == self.bound_model:
                    continue
                layer = LayerInfo(name, module)
                config = self._select_config(layer)
                if config is not None:
                    self._modules_to_compress.append((layer, config))
        return self._modules_to_compress

    def _select_config(self, layer: LayerInfo) -> Optional[Dict]:
        """
        Find the configuration for `layer` by parsing `self.config_list`.

        Parameters
        ----------
        layer
            The layer that need to check if has compression configuration.

        Returns
        -------
        Optional[Dict]
            The retrieved configuration for this layer, if None, this layer should not be compressed.
        """
        ret = None
        for config in self.config_list:
            config = config.copy()
            # expand config if key `default` is in config['op_types']
            if 'op_types' in config and 'default' in config['op_types']:
                expanded_op_types = []
                for op_type in config['op_types']:
                    if op_type == 'default':
                        expanded_op_types.extend(weighted_modules)
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

    def get_modules_wrapper(self) -> Dict[str, Module]:
        """
        Returns
        -------
        OrderedDict[str, Module]
            An ordered dict, key is the name of the module, value is the wrapper of the module.
        """
        return self.modules_wrapper

    def _wrap_model(self):
        """
        Wrap all modules that needed to be compressed.
        """
        if not self.is_wrapped:
            for _, wrapper in reversed(self.get_modules_wrapper().items()):
                _setattr(self.bound_model, wrapper.name, wrapper)
            self.is_wrapped = True

    def _unwrap_model(self):
        """
        Unwrap all modules that needed to be compressed.
        """
        if self.is_wrapped:
            for _, wrapper in self.get_modules_wrapper().items():
                _setattr(self.bound_model, wrapper.name, wrapper.module)
            self.is_wrapped = False

    def set_wrappers_attribute(self, name: str, value: Any):
        """
        To register attributes used in wrapped module's forward method.
        If the type of the value is Torch.tensor, then this value is registered as a buffer in wrapper,
        which will be saved by model.state_dict. Otherwise, this value is just a regular variable in wrapper.

        Parameters
        ----------
        name
            Name of the variable.
        value
            Value of the variable.
        """
        for wrapper in self.get_modules_wrapper():
            if isinstance(value, torch.Tensor):
                wrapper.register_buffer(name, value.clone())
            else:
                setattr(wrapper, name, value)

    def generate_graph(self, dummy_input: Any) -> TorchModuleGraph:
        """
        Generate a `TorchModuleGraph` instance of `self.bound_model` based on `jit.trace`.

        Parameters
        ----------
        dummy_input
            The dummy input for `jit.trace`, users should put it on right device before pass in.

        Returns
        -------
        TorchModuleGraph
            A `TorchModuleGraph` instance.
        """
        self._unwrap_model()
        graph = TorchModuleGraph(model=self.bound_model, dummy_input=dummy_input)
        self._wrap_model()
        return graph

    def generate_module_groups(self) -> Dict[int, List[str]]:
        """
        Get all module names in each config in config_list.

        Returns
        -------
        Dict[int, List[str]]
            A dict. The key is the config idx in config_list, the value is the module name list. i.e., {1: ['layer.0', 'layer.2']}.
        """
        self._unwrap_model()

        module_groups = {}
        for name, module in self.bound_model.named_modules():
            if module == self.bound_model:
                continue
            layer = LayerInfo(name, module)
            ret = None
            for idx, config in enumerate(self.config_list):
                config = config.copy()
                # expand config if key `default` is in config['op_types']
                if 'op_types' in config and 'default' in config['op_types']:
                    expanded_op_types = []
                    for op_type in config['op_types']:
                        if op_type == 'default':
                            expanded_op_types.extend(weighted_modules)
                        else:
                            expanded_op_types.append(op_type)
                    config['op_types'] = expanded_op_types
                # check if condition is satisified
                if 'op_types' in config and layer.type not in config['op_types']:
                    continue
                if 'op_names' in config and layer.name not in config['op_names']:
                    continue
                ret = (idx, config)
            if ret is not None and 'exclude' not in ret[1]:
                module_groups.setdefault(ret[0], [])
                module_groups[ret[0]].append(name)

        self._wrap_model()
        return module_groups

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
            parameter_name_map = {name: wrapped_param_names[id(param)] for name, param in self.bound_model.named_parameters()}
            self._wrap_model()
            return parameter_name_map
        else:
            raise Exception('When only the model is wrapped can get the parameter_name_map.')

    def _wrap_modules(self, layer: LayerInfo, config: Dict):
        """
        This method is implemented in the subclasses, i.e., `Pruner` and `Quantizer`

        Parameters
        ----------
        layer
            the layer to instrument the compression operation
        config
            the configuration for compressing this layer
        """
        raise NotImplementedError()

    def validate_config(self, model: Module, config_list: List[Dict]):
        """
        Subclass can optionally implement this method to check if config_list is valid.

        Parameters
        ----------
        model
            The model under compressed.
        config_list
            The config list used by compressor, usually specifies the 'op_types' or 'op_names' that want to compress.
        """
        pass

    def compress(self) -> Module:
        """
        Compress the model with algorithm implemented by subclass.

        The model will be instrumented and user should never edit it after calling this method.
        `self._modules_to_compress` records all the to-be-compressed layers.

        Returns
        -------
        torch.nn.Module
            model with specified modules compressed.
        """
        return self.bound_model
