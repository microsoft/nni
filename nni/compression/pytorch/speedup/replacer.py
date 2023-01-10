# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Callable, Dict

import torch

from .infer_mask import AutoMaskInference
from ..utils.attr import get_nested_attr, has_nested_attr, set_nested_attr

_logger = logging.getLogger()


class Replacer:
    def replace_modules(self, model: torch.nn.Module, auto_inferences: Dict[str, AutoMaskInference]):
        raise NotImplementedError()


class DefaultReplacer(Replacer):
    """
    This is replacer is used to replace the leaf-module in the model.
    Leaf-module is the ``torch.nn.Module`` that contains no ``torch.nn.Module`` as its attribute.

    Parameters
    ----------
    replace_module_func_dict
        A dict of module compression function, {module_type_name: replace_func}.
        The input of replace_func is the original module and its masks, the output is the compressed module,
        (original_module, (input_mask, output_mask, weight_mask)) -> compressed_module.

        Here is an exmaple for module type name ``FakeModule`` replace function::

            def fake_module_replace(ori_module, masks):
                in_mask, out_mask, weight_mask = masks
                # prune the ori_module to a new smaller module according to the mask
                return new_small_module

            replace_module_func_dict = {'FakeModule': fake_module_replace}
    """
    def __init__(self, replace_module_func_dict: Dict[str, Callable]):
        self.replace_module_func_dict = replace_module_func_dict

    def replace_modules(self, model, auto_inferences: Dict[str, AutoMaskInference]):
        replaced_names = []
        for unique_name, auto_infer in auto_inferences.items():
            if has_nested_attr(model, unique_name):
                module = get_nested_attr(model, unique_name)
                if isinstance(module, torch.nn.Module):
                    _logger.debug("replace module %s, with class type %s", unique_name, type(module))
                    self.replace_submodule(model, unique_name, auto_infer)
                    # prevent secondary replacement
                    replaced_names.append(unique_name)
                else:
                    # Support replace function in the future
                    pass
        for name in replaced_names:
            auto_inferences.pop(name)

    def replace_submodule(self, model: torch.nn.Module, unique_name: str, auto_infer: AutoMaskInference):
        module = get_nested_attr(model, unique_name)
        _logger.info("replace module (name: %s, op_type: %s)", unique_name, type(module))
        replace_function = self.replace_module_func_dict.get(type(module).__name__, None)
        if replace_function:
            compressed_module = replace_function(module, auto_infer.get_masks())
            set_nested_attr(model, unique_name, compressed_module)
