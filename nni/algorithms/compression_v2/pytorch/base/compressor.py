import collections
from copy import deepcopy
import logging
from typing import List, Dict, Optional, OrderedDict, Tuple, Any

import torch
from torch.nn import Module
from torch.tensor import Tensor

from nni.common.graph_utils import TorchModuleGraph

_logger = logging.getLogger(__name__)


class LayerInfo:
    def __init__(self, name: str, module: Module):
        self.module = module
        self.name = name
        self.type = type(module).__name__


def _setattr(model: Module, name: str, module: Module):
    name_list = name.split(".")
    for name in name_list[:-1]:
        model = getattr(model, name)
    setattr(model, name_list[-1], module)


weighted_modules = [
    'Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d',
    'Linear', 'Bilinear',
    'PReLU',
    'Embedding', 'EmbeddingBag',
]


class Compressor:
    def __init__(self, model: Module, config_list: List[Dict], back_up: bool, graph_needed: bool = False, dummy_input: Optional[Tensor] = None, **kwargs):
        """
        Parameters
        ----------
        model
            The model under compressed.
        config_list
            The config list used by compressor, usually specifies the 'op_types' or 'op_names' that want to compress.
        back_up
            Set True to save the original model and config_list for reset, set False to skip it.
        """
        assert isinstance(model, Module)

        self._back_up = back_up
        self._graph_needed = graph_needed
        self._dummy_input = dummy_input
        self.is_wrapped = False

        self._origin_model = None
        self._origin_config_list = None

        self.reset(model=model, config_list=config_list)

    def reset(self, model: Optional[Module] = None, config_list: Optional[List[Dict]] = None):
        if not self._back_up:
            assert model is not None and config_list is not None, 'Must set model and config_list to reset an un-backup compressor.'
            self.bound_model = model
            self.config_list = config_list
        else:
            if model is None:
                self.bound_model = deepcopy(self._origin_model)
            else:
                self.bound_model = model
                self._origin_model = deepcopy(model)
            if config_list is None:
                self.config_list = deepcopy(self._origin_config_list)
            else:
                self.config_list = config_list
                self._origin_config_list = deepcopy(config_list)
        self.validate_config(model=model, config_list=config_list)

        self._unwrap_model()

        self._modules_to_compress = None
        self.config_based_group_info = None
        self.modules_wrapper = collections.OrderedDict()
        for layer, config in self._detect_modules_to_compress():
            wrapper = self._wrap_modules(layer, config)
            self.modules_wrapper[layer.name] = wrapper

        self.graph = None
        if self._graph_needed:
            assert self._dummy_input is not None, 'dummy_input is needed for generate graph.'
            self.graph = TorchModuleGraph(model=self.bound_model, dummy_input=self._dummy_input)

        self._wrap_model()

    def _detect_modules_to_compress(self) -> List[Tuple[LayerInfo, Dict]]:
        """
        Detect all modules should be compressed, and save the result in `self._modules_to_compress`.
        The model will be instrumented and user should never edit it after calling this method.
        """
        if self._modules_to_compress is None:
            self._modules_to_compress = []
            self.config_based_group_info = {}
            for name, module in self.bound_model.named_modules():
                if module == self.bound_model:
                    continue
                layer = LayerInfo(name, module)
                result = self._select_config(layer)
                if result is not None:
                    idx, config = result
                    self._modules_to_compress.append((layer, config))
                    self.config_based_group_info.setdefault(idx, [])
                    self.config_based_group_info[idx].append(layer.name)
        return self._modules_to_compress

    def _select_config(self, layer: LayerInfo) -> Optional[Tuple[int, Dict]]:
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
        if ret is None or 'exclude' in ret:
            return None
        return ret

    def _get_modules_wrapper(self) -> OrderedDict[str, Module]:
        return self.modules_wrapper

    def _wrap_model(self):
        """
        Wrap all modules that needed to be compressed.
        """
        for _, wrapper in reversed(self._get_modules_wrapper().items()):
            _setattr(self.bound_model, wrapper.name, wrapper)
        self.is_wrapped = True

    def _unwrap_model(self):
        """
        Unwrap all modules that needed to be compressed.
        """
        if self.is_wrapped:
            for _, wrapper in self._get_modules_wrapper().items():
                _setattr(self.bound_model, wrapper.name, wrapper.module)
            self.is_wrapped = False

    def set_wrappers_attribute(self, name: str, value: Any):
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
