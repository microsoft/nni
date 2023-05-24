# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import TYPE_CHECKING, Callable, Dict

from torch.utils._pytree import tree_map

from ..utils import get_nested_attr, set_nested_attr

if TYPE_CHECKING:
    from .model_speedup import ModelSpeedup

_logger = logging.getLogger(__name__)


class Replacer:
    def replace_modules(self, speedup: 'ModelSpeedup'):
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

    def replace_modules(self, speedup: 'ModelSpeedup'):
        visited = set()
        for node, node_info in speedup.node_infos.items():
            if node.op == 'call_module':
                if node.target in visited:
                    node_info.replaced = True
                    continue
                visited.add(node.target)
                module = get_nested_attr(speedup.bound_model, node.target)
                module_type = type(module).__name__
                replace_function = self.replace_module_func_dict.get(module_type, None)
                if replace_function:
                    _logger.info("replace module (name: %s, op_type: %s)", node.name, module_type)
                    assert len(node.kwargs) == 0
                    in_masks = tree_map(lambda n: speedup.node_infos[n].output_masks, node.args)
                    compressed_module = replace_function(module, (in_masks, node_info.output_masks, node_info.param_masks))
                    set_nested_attr(speedup.bound_model, node.target, compressed_module)
                    node_info.replaced = True
            else:
                pass
