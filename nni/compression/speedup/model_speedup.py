# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import inspect
import logging
from pathlib import Path
import tempfile
from typing import Any, Dict, List

import torch
import torch.fx
from torch.fx import GraphModule
from torch.fx._compatibility import compatibility
from torch.fx.node import Node, Target
from torch.fx.passes.shape_prop import ShapeProp

from nni.common.concrete_trace_utils import concrete_trace
from nni.compression.utils import set_nested_attr
from nni.compression.speedup.replacement import replace_module

from .container import NodeInfo
from .mask_conflict import fix_channel_mask_conflict, fix_group_mask_conflict, fix_weight_sharing_mask_conflict
from .mask_updater import (MaskUpdater,
                           DefaultMaskUpdater,
                           LeafModuleMaskUpdater,
                           NoMaskUpdater,
                           NoChangeMaskUpdater)
from .replacement import replace_module
from .replacer import Replacer, DefaultReplacer
from .utils import tree_map_zip, poss_deepcopy, randomize_like_with_shape, torch_integer_dtype
from ..utils import set_nested_attr


def _normalize_input(dummy_input: Any) -> Any:
    if isinstance(dummy_input, torch.Tensor):
        dummy_input = (dummy_input, )
    elif isinstance(dummy_input, list):
        dummy_input = tuple(dummy_input)
    elif isinstance(dummy_input, dict):
        dummy_input = tuple(dummy_input.values())
    return dummy_input


def sparsity_stats(mask: Dict[str, torch.Tensor]) -> str:
    """
    Calculate the sparsity of a mask.

    Parameters
    ----------
    mask
        The mask tensor.

    Returns
    -------
    str
        The sparsity of the mask.
    """
    ret = ''
    for k, v in mask.items():
        if isinstance(v, torch.Tensor):
            ret += f'{k}: {1 - v.nonzero().size(0) / v.numel(): .4f} '
    return ret

@compatibility(is_backward_compatible=True)
class ModelSpeedup(torch.fx.Interpreter):
    """
    This class is to speedup the model with provided weight mask, the masked module will be replaced by a new dense module.
    ModelSpeedup use concrete trace based on ``torch.fx`` to get the graph,
    note that the trace may fail if there is stochastic structure in the model.

    Parameters
    ----------
    model
        The model user wants to speedup.
    dummy_input
        A tensor or a tuple, the dummy input to execute the model.
    masks_or_file
        The path of user provided masks file, or the masks object.
    map_location
        The device on which masks are placed, same to map_location in ```torch.load```.
    batch_dim
        The index of batch dimension in the dummy_input.
    batch_size
        The batch_size coefficient of the sparsity inference.
        This value is actually used as the batchsize of the dummy_input.
    customized_mask_updaters
        A list of ``MaskUpdater``.
        NNI will automatically infer sparsity based on the data distribution in the forward and backward process,
        but if some special operations lead to automatic sparsity inference errors,
        users can manually set the mask inference rules for the special operations to make the mask inference correct.
    customized_replacers
        A list of ``Replacer``.
        The replacer is used to replace the origin module with a compressed module.
        Users can costomized the replacement logic by customized a replacer.
        Before the built-in replacement logic in nni is executed,
        the replacement logic in the customized replacer list will be executed sequentially first.
    graph_module
        A torch.fx.GraphModule.
        If ModelSpeedup default concrete trace cannot meet the needs,
        users can directly pass in a torch.fx.GraphModule instead.
    garbage_collect_values
        If the garbage_collect_values is True, nni will delete cache information after the cache has none usage.
    logger
        Set a logger. If the value is None, nni will use the default logger.
    """
    STD_DELTA = 1e-6

    def __init__(self,
                 model: torch.nn.Module,
                 dummy_input: Any,
                 masks_or_file: Any,
                 map_location: Any = None,
                 batch_dim: int = 0,
                 batch_size: int = 8,
                 customized_mask_updaters: List[MaskUpdater] | None = None,
                 customized_replacers: List[Replacer] | None = None,
                 graph_module: GraphModule | None = None,
                 garbage_collect_values: bool = True,
                 logger: logging.Logger | None = None):
        self.dummy_input = _normalize_input(dummy_input)
        self.bound_model = model
        if isinstance(graph_module, GraphModule):
            self.graph_module = graph_module
        elif isinstance(dummy_input, dict):
            self.graph_module = concrete_trace(model, dummy_input)
        else:
            self.graph_module = concrete_trace(model, self.dummy_input)

        ShapeProp(self.graph_module).propagate(*self.dummy_input)   # attach shape to graph_module

        super().__init__(self.graph_module, garbage_collect_values)

        if isinstance(masks_or_file, (str, Path)) and Path(masks_or_file).exists():
            self.masks = torch.load(masks_or_file, map_location)
        elif isinstance(masks_or_file, dict):
            self.masks = masks_or_file
        else:
            raise Exception('Please provide the mask or the path of the mask file.')

        self.batch_dim = batch_dim
        self.batch_size = batch_size

        self.mask_updaters: List[MaskUpdater] = [
            *(customized_mask_updaters if customized_mask_updaters else []),
            NoChangeMaskUpdater(),
            NoMaskUpdater(),
            LeafModuleMaskUpdater(),
            DefaultMaskUpdater()
        ]

        assert customized_replacers is None or all(isinstance(replacer, Replacer) for replacer in customized_replacers)
        self.replacers = customized_replacers if customized_replacers is not None else []
        self.replacers.append(DefaultReplacer(replace_module_func_dict=replace_module))

        if logger == None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
        else:
            self.logger = logger

        self.node_infos: Dict[Node, NodeInfo] = {}
        for node in self.graph_module.graph.nodes:
            self.node_infos[node] = NodeInfo(node)

    @compatibility(is_backward_compatible=True)
    def store_attr(self, path: str, obj: Any):
        set_nested_attr(self.graph_module, path, obj)

    @compatibility(is_backward_compatible=True)
    def placeholder(self, target: Target, args, kwargs) -> Any:
        """
        Override the execution for 'placeholder' ops.
        """
        return self.arg_dict[target]

    def tensor_propagate_check(self, obj: torch.Tensor):
        """
        Detect the tensor should be seen as an intermediate tensor.
        """
        return obj.numel() > self.batch_size and obj.numel() % self.batch_size == 0

    def direct_calc_mask(self, output: Any, output_mask: torch.Tensor | None = None, batch_dim: int | None = None):
        batch_dim = self.batch_dim if batch_dim is None else batch_dim
        if isinstance(output, torch.Tensor) and self.tensor_propagate_check(output):
            mask_size = list(output.size())
            mask_size[batch_dim] = 1
            output_mask = torch.ones(mask_size).type_as(output).float() if output_mask is None else output_mask.clone()
            output: torch.Tensor = output.transpose(0, batch_dim)
            output_mask = output_mask.transpose(0, batch_dim)
            if output.dtype in torch_integer_dtype:
                same = output[:] == output[0]
                reduced = torch.sum(same, dim=0)
                is_constant = reduced == output.size(0)
                output_mask[:, is_constant] = 0.
            else:
                std = torch.std(output, dim=0)
                mask_pos = std < self.STD_DELTA
                output_mask[:, mask_pos] = 0.
            return output_mask.transpose(0, batch_dim)
        else:
            return None

    def indirect_calc_mask(self, output_grad: torch.Tensor, output_mask: torch.Tensor, batch_dim: int | None = None):
        batch_dim = self.batch_dim if batch_dim is None else batch_dim
        if isinstance(output_grad, torch.Tensor) and self.tensor_propagate_check(output_grad):
            output_grad = output_grad.transpose(0, batch_dim)
            output_mask = output_mask.clone().transpose(0, batch_dim)
            assert output_grad.shape[1:] == output_mask.shape[1:]
            gradient_sum = torch.sum(torch.abs(output_grad), dim=0)
            _grad_zero = gradient_sum == 0.
            output_mask[:, _grad_zero] = 0.
            return output_mask.transpose(0, batch_dim)
        return output_mask

    # backward the output grad_fn with output_mask as grad
    def indirect_backward(self, output: Any, output_mask: torch.Tensor | None):
        if isinstance(output, torch.Tensor) and self.tensor_propagate_check(output):
            assert isinstance(output_mask, torch.Tensor)
            if output.grad_fn is not None:
                output.backward(output_mask.expand_as(output))
        else:
            assert output_mask is None

    # pass the gradient to the predecessor nodes
    def indirect_pass_grad(self, node: Node, outputs: Any):
        def add_grad(grad, output):
            if isinstance(output, torch.Tensor):
                if grad is not None and output.grad is not None:
                    return grad + output.grad
                elif grad is None:
                    return output.grad
                else:
                    return grad
            else:
                return grad

        self.node_infos[node].output_grad = tree_map_zip(add_grad, self.node_infos[node].output_grad, outputs)

    def fix_mask_conflict(self):
        fix_group_mask_conflict(self.graph_module, self.masks)
        fix_channel_mask_conflict(self.graph_module, self.masks)
        fix_weight_sharing_mask_conflict(self.graph_module, self.masks)


    def propagate_originally(self):
        """
        Propagate normally to get informations of intermediate variables such as shape, dtype of tensors.
        Default action: execute and store output to node_info.output_origin(intermediate variables when assigned),
        and node_info.output_inplace(intermediate variables after in-place ops).
        """
        self.logger.info("Propagate original variables")
        for node in self.graph_module.graph.nodes:
            node: Node

            args, kwargs = node.args, node.kwargs
            args = tree_map_zip(lambda nd: self.node_infos[nd].output_inplace if isinstance(nd, Node) else nd, args)
            kwargs = tree_map_zip(lambda nd: self.node_infos[nd].output_inplace if isinstance(nd, Node) else nd, kwargs)
            output = getattr(self, node.op)(node.target, args, kwargs)

            self.node_infos[node].output_origin = output
            self.node_infos[node].output_inplace = \
                tree_map_zip(lambda t: t.clone().detach() if isinstance(t, torch.Tensor) else poss_deepcopy(t, self.logger), output)
            self.node_infos[node].output_masks = \
                tree_map_zip(lambda t: torch.ones_like(t).clone().detach() if isinstance(t, torch.Tensor) else None, output)

            sp = f', {sparsity_stats(self.masks.get(node.target, {}))}' if node.op == 'call_module' else ''
            sp += f', {sparsity_stats({"output mask": self.node_infos[node].output_masks})}'
            self.logger.info('Propagate variables for %s: %s%s', node.op, node.name, sp)

            if self.garbage_collect_values:
                # do memory collect to reduce memory usage
                for to_delete in self.user_to_last_uses.get(node, []):
                    del self.node_infos[to_delete].output_inplace

    def update_direct_sparsity(self):
        # update direct out mask
        self.logger.info("Update direct sparsity...")

        for node in self.graph_module.graph.nodes:
            node: Node
            self.node_infos[node].mask_updater.direct_update_preprocess(self, node)

        for node in self.graph_module.graph.nodes:
            node: Node
            self.node_infos[node].mask_updater.direct_update_process(self, node)
            sp = f', {sparsity_stats(self.masks.get(node.target, {}))}' if node.op == 'call_module' else ''
            sp += f', {sparsity_stats({"output mask": self.node_infos[node].output_masks})}'
            self.logger.info('Update direct mask for %s: %s%s', node.op, node.name, sp)

        for node in self.graph_module.graph.nodes:
            node: Node
            self.node_infos[node].mask_updater.direct_update_postprocess(self, node)

    def update_indirect_sparsity(self):
        # update indirect out mask
        self.logger.info("Update indirect sparsity...")

        for node in reversed(self.graph_module.graph.nodes):
            node: Node
            self.node_infos[node].mask_updater.indirect_update_preprocess(self, node)

        for node in reversed(self.graph_module.graph.nodes):
            node: Node
            self.node_infos[node].mask_updater.indirect_update_process(self, node)
            sp = f', {sparsity_stats(self.masks.get(node.target, {}))}' if node.op == 'call_module' else ''
            sp += f', {sparsity_stats({"output mask": self.node_infos[node].output_masks})}'
            self.logger.info('Update indirect mask for %s: %s%s', node.op, node.name, sp)

        for node in reversed(self.graph_module.graph.nodes):
            node: Node
            self.node_infos[node].mask_updater.indirect_update_postprocess(self, node)

    def replace_compressed_modules(self):
        """
        Replace all the modules that have changed (weights/inputs/output) shape.
        The new module is created using the same arguments of the to-be-replaced module,
        and correctly inherits its weights.

        NOTE: ```func``` type cannot be replaced as it is not a module, thus, one limitation
        is that ```func``` should be not required to be replaced.
        """
        self.logger.info("Replace compressed modules...")
        # the mask conflict should be already resolved
        with torch.no_grad():
            for replacer in self.replacers:
                replacer.replace_modules(self)
        for node in self.node_infos:
            if node.op == 'call_module' and not self.node_infos[node].replaced:
                module = self.fetch_attr(node.target)
                module_type = module._get_name()
                err_msg = f"Has not supported replacing module with type: {module_type}, "
                err_msg += f"you could report an issue at https://github.com/microsoft/nni. "
                err_msg += f"If you know how to replace {module_type}, "
                err_msg += f"you could implement module replacement by passing in"
                err_msg += f"`customized_replacers` to `{self.__class__.__name__}`. "
                err_msg += f"You are welcome to contribute back to nni as native support "
                err_msg += f"if you have implemented the replacement function, "
                err_msg += f"so that more users can benefit from your contributions."
                self.logger.error(err_msg)

    def initialize_propagate(self, args):
        def model_tensor_randomizer(obj):
            if isinstance(obj, torch.Tensor) and obj.dim() > self.batch_dim:
                input_shape = list(obj.size())
                # set the batchsize to the confidence ratio
                input_shape[self.batch_dim] = self.batch_size
                return randomize_like_with_shape(input_shape, obj)
            else:
                return obj

        # input of the whole model
        placeholders: List[Node] = [node for node in self.graph_module.graph.nodes if node.op == 'placeholder']
        assert len(args) <= len(placeholders)
        args = tree_map_zip(model_tensor_randomizer, args)
        self.arg_dict = {}
        for i, placeholder in enumerate(placeholders):
            if i < len(args):
                self.arg_dict[placeholder.target] = args[i]
            else:
                assert len(placeholder.args) == 1, f'Parameter \'{placeholder.target}\' has no default value!'
                self.arg_dict[placeholder.target] = placeholder.args[0]

    def initialize_update_sparsity(self):
        # for mask_updater to store extended infos
        for node in self.node_infos:
            for mask_updater in self.mask_updaters:
                if mask_updater.detect(self, node):
                    self.node_infos[node].mask_updater = mask_updater
                    break

        for node_info in self.node_infos.values():
            if node_info.module is None:
                continue
            masks = self.masks.get(node_info.node.target, {})

            output_masks = {name: masks[name] for name in filter(lambda name: name.startswith('_output_'), masks.keys())}
            if output_masks:
                if isinstance(node_info.output_masks, torch.Tensor):
                    node_info.output_masks *= list(output_masks.values())[0]
                elif isinstance(node_info.output_masks, (list, tuple)):
                    for key, mask in output_masks.items():
                        key = key.split('_output_')[1]
                        assert key.isnumeric()
                        if mask is not None:
                            node_info.output_masks[int(key)] *= mask
                elif isinstance(node_info.output_masks, dict):
                    for key, mask in output_masks.items():
                        if mask is not None:
                            key = key.split('_output_')[1]
                            node_info.output_masks[key] *= mask
                else:
                    raise RuntimeError(f'Unsupported output type {type(node_info.output_masks)}.')

            input_masks = {name: masks[name] for name in filter(lambda name: name.startswith('_input_'), masks.keys())}
            if input_masks:
                func = self.fetch_attr(node_info.node.target).forward
                while hasattr(func, '__wrapped__'):
                    func = func.__wrapped__
                arg_list = inspect.getfullargspec(func).args
                kw_to_posi = dict(zip(arg_list[1:], range(len(arg_list) - 1)))
                node_kw = {
                    **dict(zip(range(len(arg_list) - 1), node_info.node.args)),
                    **dict(zip(arg_list[1:], node_info.node.args)),
                    **{kw_to_posi[k]: v for k, v in node.kwargs.items()},
                    **node_info.node.kwargs,
                }
                for key, mask in input_masks.items():
                    key = key.split('_input_')[1]
                    key = int(key) if key.isnumeric() else key
                    if isinstance(mask, torch.Tensor):
                        assert isinstance(self.node_infos[node_kw[key]].output_masks, torch.Tensor)
                        self.node_infos[node_kw[key]].output_masks *= mask.detach().clone()

    def speedup_model(self) -> torch.nn.Module:
        try:
            ori_state_dict_file = tempfile.NamedTemporaryFile(delete=False)
            torch.save(self.graph_module.state_dict(), ori_state_dict_file)
            ori_state_dict_file.close()

            self.logger.info("Start to speedup the model...")
            training = self.graph_module.training
            self.graph_module.train(False)

            # TODO: suppose to fix the conflict after the sparsity propagation, which is more elegent
            self.logger.info('Resolve the mask conflict before mask propagate...')
            # fix_mask_conflict(self.masks, self.graph_module, self.dummy_input)
            self.fix_mask_conflict()
            self.logger.info('Infer module masks...')
            self.initialize_propagate(self.dummy_input)
            self.propagate_originally()
            self.initialize_update_sparsity()
            self.update_direct_sparsity()
            self.update_indirect_sparsity()
            self.logger.info('Resolve the mask conflict after mask propagate...')
            # fix_mask_conflict(self.masks, self.graph_module, self.dummy_input)
            self.fix_mask_conflict()

            self.graph_module.load_state_dict(torch.load(ori_state_dict_file.name))
            self.graph_module.train(training)
        finally:
            import os
            os.unlink(ori_state_dict_file.name)

        self.replace_compressed_modules()
        self.logger.info("Speedup done.")

        return self.bound_model

    def run(self):
        return self.speedup_model()
