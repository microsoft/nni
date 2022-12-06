# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import logging
import queue
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, Union

import torch
import torch.fx
import torch.nn as nn
from torch.fx import GraphModule
from torch.fx._compatibility import compatibility
from torch.fx.node import Argument, Node, Target

from nni.common.concrete_trace_utils.utils import (map_recursive,
                                                   map_recursive_zip,
                                                   run_onlyif_instance)
from nni.compression.pytorch.speedup.compress_modules import replace_module
from nni.compression.pytorch.speedup.v2.container import NodeInfo, Slot
from nni.compression.pytorch.speedup.v2.mask_updater import MaskUpdater, DefaultMaskUpdater, DefaultModuleMaskUpdater
from nni.compression.pytorch.utils.mask_conflict import fix_mask_conflict
from nni.compression.pytorch.utils.utils import (rand_like_with_shape,
                                                 randomize_tensor,
                                                 torch_float_dtype,
                                                 torch_integer_dtype)


@compatibility(is_backward_compatible=True)
class ModelSpeedup(torch.fx.Interpreter):
    """
    This class is to speedup the model with provided weight mask.

    Parameters
    ----------
    model : pytorch model
        The model user wants to speedup
    masks_file : str/dict
        The path of user provided mask file, or the mask object
    map_location : str
        the device on which masks are placed, same to map_location in ```torch.load```
    batch_dim : int
        the index of batch dimension in the dummy_input
    batch_size: the batch_size coefficient of the sparsity inference. This value is
        actually used as the batchsize of the dummy_input.
    customized_replace_func: None/Dict
        If `customized_replace_func` is not None, then we will use the given function to replace the
        corresponding modules. The `key` of the dict is the opertor types and the `value`
        is the replace function of corresponding opertor. The replace function should take
        two input parameters, one is the original module, the second input parameter is tuple
        of the input mask, output mask and weight mask. This replace function should prune the module
        accordingly. Here is an example of the replace function(more examples can refer to compress_modules.py)::

            def example_replace(ori_module, masks):
                in_mask, out_mask, weight_mask = masks
                # prune the ori_module to a new smaller module according to the mask
                return new_small_module
    garbage_collect_values: bool
        if the garbage_collect_values is True, we will delete cache information after the cache has none usage.
    logger: Optional[logging.Logger]
        to set logger. if the value is None we will use the default logger
    """
    STD_DELTA = 1e-6
    randomize_range_float = (0.1, 8.0)

    def __init__(self,
                module: GraphModule,
                batch_dim=0,
                batch_size=8,
                customized_replace_func=dict(),
                customized_mask_updaters: list[MaskUpdater]=[],
                garbage_collect_values: bool=True,
                logger:Optional[logging.Logger]=None):
        super().__init__(module, garbage_collect_values)

        self.module: GraphModule
        self.batch_dim = batch_dim
        self.batch_size = batch_size
        self.customized_replace_func = customized_replace_func
        self.mask_updaters:list[MaskUpdater] = [
            *customized_mask_updaters,
            DefaultModuleMaskUpdater(),
            DefaultMaskUpdater()
        ]
        if logger == None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
        else:
            self.logger = logger

    @compatibility(is_backward_compatible=True)
    def store_attr(self, path: str, obj):
        """
        an util for interprete and edit the module.

        Parameters
        ---------
        path: str
            the path for the attr. the hierarchy is split by `.`.
        obj:
            The target node to store.
        """
        target_atoms = path.split('.')
        attr_itr = self.module
        for i in range(len(target_atoms))[:-1]:
            atom = target_atoms[i]
            if not hasattr(attr_itr, atom):
                raise RuntimeError(f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}")
            attr_itr = getattr(attr_itr, atom)
        setattr(attr_itr, target_atoms[-1], obj)

    @compatibility(is_backward_compatible=True)
    def placeholder(self, target, args, kwargs) -> Any:
        """
        override the execution for 'placeholder' ops.
        """
        return self.arg_dict[target]

    def tensor_propagate_check(self, obj: torch.Tensor):
        """
        detect the tensor should be seen as an intermediate tensor.
        sometimes
        """
        return obj.numel() > self.batch_size and obj.numel() % self.batch_size == 0

    @run_onlyif_instance(torch.Tensor)
    def tensor_cloner(self, obj: torch.Tensor):
        return obj.clone()

    @run_onlyif_instance(torch.Tensor)
    def tensor_detacher(self, obj: torch.Tensor):
        return obj.detach()

    @run_onlyif_instance(torch.Tensor)
    def tensor_clone_detacher(self, obj: torch.Tensor):
        return obj.clone().detach()

    @run_onlyif_instance(Node)
    def slot_getter_value_0(self, node: Node):
        assert self.slots[node].status['value_0'] == 1, 'slot error: bad value_0(%d)' % self.slots[node].status['value_0']
        return self.slots[node].value_0

    @run_onlyif_instance(Node)
    def slot_getter_value_1(self, node: Node):
        assert self.slots[node].status['value_1'] == 1, 'slot error: bad value_1(%d)' % self.slots[node].status['value_1']
        return self.slots[node].value_1

    @run_onlyif_instance(Node)
    def slot_getter_value_2(self, node: Node):
        assert self.slots[node].status['value_2'] == 1, 'slot error: bad value_2(%d)' % self.slots[node].status['value_2']
        return self.slots[node].value_2

    def tensor_randomizer(self, obj):
        if isinstance(obj, torch.Tensor):
            if obj.numel() != 1 and len(obj.size()) > self.batch_dim and obj.size(self.batch_dim) == self.batch_size:
                new_obj = obj.clone().detach()
                if not new_obj.is_contiguous():
                    new_obj = new_obj.contiguous()
                randomize_tensor(new_obj, self.randomize_range_float[0], self.randomize_range_float[1])
                return new_obj
            else:
                new_obj = obj.clone().detach()
                return new_obj
        else:
            try:
                return copy.deepcopy(obj)
            except copy.Error:
                return obj

    @run_onlyif_instance(Node, False)
    def slot_getter_mask_1(self, node: Node):
        assert self.slots[node].status['mask_1'] == 1, 'slot error: bad mask_1(%d)' % self.slots[node].status['mask_1']
        return self.slots[node].mask_1

    @run_onlyif_instance(Node, False)
    def slot_getter_mask_2(self, node: Node):
        if self.slots[node].mask_2 is None:
            return None
        else:
            assert self.slots[node].status['mask_2'] >= 1, 'slot error: bad mask_2(%d)' % self.slots[node].status['mask_2']
            return self.slots[node].mask_2

    @run_onlyif_instance(Node, False)
    def slot_getter_mask_2_or_1(self, node: Node):
        if self.slots[node].mask_2 is None:
            assert self.slots[node].status['mask_1'] == 1, 'slot error: bad mask_1(%d)' % self.slots[node].status['mask_1']
            return self.slots[node].mask_1
        else:
            assert self.slots[node].status['mask_2'] >= 1, 'slot error: bad mask_2(%d)' % self.slots[node].status['mask_2']
            return self.slots[node].mask_2

    def mask_applier(self, value, mask):
        if isinstance(value, torch.Tensor) and self.tensor_propagate_check(value):
            assert isinstance(mask, torch.Tensor) and value.shape == mask.shape
            return value * mask
        else:
            assert mask is None
            return value

    def tensor_requires_grad(self, obj):
        if isinstance(obj, torch.Tensor) and self.tensor_propagate_check(obj) and obj.dtype in torch_float_dtype:
            # only float type can require the gradient
            # enable the auto gradient
            obj.requires_grad_(True)

    # utils for direct update tand indirect update
    def direct_calc_mask(self, obj):
        if isinstance(obj, torch.Tensor) and self.tensor_propagate_check(obj):
            obj: torch.Tensor

            out_mask = torch.ones_like(obj)
            if obj.dtype in torch_integer_dtype:
                same = obj[:] == obj[0]
                reduced = torch.sum(same, dim=0)
                is_constant = reduced == obj.size(0)
                out_mask[:, is_constant] = 0
            else:
                std = torch.std(obj, dim=0)
                mask_pos = std < self.STD_DELTA
                out_mask[:, mask_pos] = 0
            return out_mask
        else:
            return None

    def indirect_calc_mask(self, mask, obj):
        if isinstance(obj, torch.Tensor) and self.tensor_propagate_check(obj):
            assert isinstance(mask, torch.Tensor) and obj.shape == mask.shape
            if obj.grad is not None:
                gradient_sum = torch.sum(torch.abs(obj.grad), dim=0)
                _grad_zero = gradient_sum == 0
                new_mask = mask.clone()
                for batchid in range(obj.size(0)):
                    # set the same mask value for the whole batche
                    new_mask[batchid][_grad_zero] = 0
                return new_mask
        return mask

    def indirect_update_param_mask(self, output, mask):
        # Note: output maybe tensor or list/tuple of tensors
        if isinstance(output, torch.Tensor) and self.tensor_propagate_check(output):
            assert isinstance(mask, torch.Tensor)
            if output.grad_fn is not None:
                output.backward(mask)
        else:
            assert not isinstance(mask, torch.Tensor)

    # # pass the gradient to the predecessor nodes
    def indirect_pass_grad(self, slot_val, out):
        if isinstance(slot_val, torch.Tensor):
            assert isinstance(out, torch.Tensor)
            if self.tensor_propagate_check(slot_val):
                if slot_val.grad is not None and out.grad is not None:
                    slot_val.grad.data += out.grad.data
                elif slot_val.grad is None:
                    slot_val.grad = out.grad
                elif slot_val.grad is not None and out.grad is None:
                    # for example, tin.view(batch, tin.size(1)/2, tin.view(2)*2)
                    # the size operation of tin will have no gradient
                    pass
        else:
            assert not isinstance(out, torch.Tensor)

    def propagate_originally(self):
        self.logger.info("propagate original variables")
        for node in self.module.graph.nodes:
            node: Node

            self.logger.info('Propagate variables for %s: %s', node.op, node.name)

            args, kwargs = node.args, node.kwargs
            args = map_recursive(self.slot_getter_value_1, args)
            args = map_recursive(self.tensor_detacher, args)
            kwargs = map_recursive(self.slot_getter_value_1, kwargs)
            kwargs = map_recursive(self.tensor_detacher, kwargs)

            output = getattr(self, node.op)(node.target, args, kwargs)

            self.slots[node].value_0 = output
            self.slots[node].status['value_0'] += 1
            self.slots[node].value_1 = map_recursive(self.tensor_clone_detacher, output)
            self.slots[node].status['value_1'] += 1

            if self.garbage_collect_values:
                # TODO: do memory collect / compression
                # for to_delete in self.user_to_last_uses.get(node, []):
                #     self.slots[node].value_1 = None
                pass

    def update_direct_sparsity(self):
        # update indirect out mask

        self.logger.info("update direct sparsity")

        for node in self.module.graph.nodes:
            node: Node
            self.node_infos[node].mask_updater.direct_update_preprocess(self, node)

        for node in self.module.graph.nodes:
            node: Node
            self.logger.info('Update direct mask for %s: %s', node.op, node.name)
            self.node_infos[node].mask_updater.direct_update_process(self, node)

        for node in self.module.graph.nodes:
            node: Node
            self.node_infos[node].mask_updater.direct_update_postprocess(self, node)

        return
        for node in self.module.graph.nodes:
            node: Node

            self.slots[node].value_2 = map_recursive(self.tensor_randomizer, self.slots[node].value_0)
            self.slots[node].status['value_2'] += 1

        for node in self.module.graph.nodes:
            node: Node

            self.logger.info('Update direct mask for %s: %s', node.op, node.name)
            with torch.no_grad():
                if node.op == 'call_module':
                    node_info: NodeInfo = self.node_infos[node]
                    sub_module: nn.Module = self.fetch_attr(node.target)

                    for _k, v in sub_module.named_parameters():
                        randomize_tensor(v, self.randomize_range_float[0], self.randomize_range_float[1])

                    for k, v in sub_module.named_parameters():
                        v *= node_info.param_masks_0[k] # in-place addition

                args = map_recursive(self.slot_getter_value_2, node.args)
                arg_masks = map_recursive(self.slot_getter_mask_1, node.args)
                args = map_recursive_zip(self.mask_applier, args, arg_masks)
                kwargs = map_recursive(self.slot_getter_value_2, node.kwargs)
                kwarg_masks = map_recursive(self.slot_getter_mask_1, node.kwargs)
                kwargs = map_recursive_zip(self.mask_applier, kwargs, kwarg_masks)

                output = getattr(self, node.op)(node.target, args, kwargs)

                self.slots[node].mask_1 = map_recursive(self.direct_calc_mask, output)
                self.slots[node].status['mask_1'] += 1

            # do memory collect / compression

    def update_indirect_sparsity(self):
        # update indirect out mask

        self.logger.info("update indirect sparsity")


        for node in reversed(self.module.graph.nodes):
            node: Node
            self.node_infos[node].mask_updater.indirect_update_preprocess(self, node)

        for node in reversed(self.module.graph.nodes):
            node: Node
            self.logger.info('Update indirect mask for %s: %s', node.op, node.name)
            self.node_infos[node].mask_updater.indirect_update_process(self, node)

        for node in reversed(self.module.graph.nodes):
            node: Node
            self.node_infos[node].mask_updater.indirect_update_postprocess(self, node)

        return
        for node in reversed(self.module.graph.nodes):
            node: Node

            self.logger.info('Update indirect mask for %s: %s', node.op, node.name)

            output = map_recursive(self.slot_getter_value_2, node)
            output_masks_1 = map_recursive(self.slot_getter_mask_1, node)
            output_masks_2 = map_recursive_zip(self.indirect_calc_mask, output_masks_1, output)

            self.slots[node].mask_2 = output_masks_2
            self.slots[node].status['mask_2'] += 1

            # init apply input
            # randomized, so it's same to use slot_getter_value_orig or slot_getter_value_orig_inplace
            args = map_recursive(self.slot_getter_value_0, node.args)
            args = map_recursive(self.tensor_randomizer, args)
            arg_masks = map_recursive(self.slot_getter_mask_1, node.args)
            args = map_recursive_zip(self.mask_applier, args, arg_masks)
            map_recursive(self.tensor_requires_grad, args)

            kwargs = map_recursive(self.slot_getter_value_0, node.kwargs)
            kwargs = map_recursive(self.tensor_randomizer, kwargs)
            kwarg_masks = map_recursive(self.slot_getter_mask_1, node.kwargs)
            kwargs = map_recursive_zip(self.mask_applier, kwargs, kwarg_masks)
            map_recursive(self.tensor_requires_grad, kwargs)

            # Some operator may have the in_place operations, so we need to clone the input
            # before passing to the self.module
            args_cloned = map_recursive(self.tensor_cloner, args)
            kwargs_cloned = map_recursive(self.tensor_cloner, kwargs)

            output = getattr(self, node.op)(node.target, args_cloned, kwargs_cloned)

            map_recursive_zip(self.indirect_update_param_mask, output, output_masks_2)

            if node.op == 'call_module':
                # update the sparsity of the paramters
                node_info: NodeInfo = self.node_infos[node]
                sub_module: nn.Module = self.fetch_attr(node.target)
                for k, v in sub_module.named_parameters():
                    grad_zero = v.grad.data == 0
                    node_info.param_masks_1[k] = node_info.param_masks_0[k].clone()
                    node_info.param_masks_1[k][grad_zero] = 0


            arg_values_2 = map_recursive(self.slot_getter_value_2, node.args)
            kwarg_values_2 = map_recursive(self.slot_getter_value_2, node.kwargs)

            map_recursive_zip(self.indirect_pass_grad, arg_values_2, args)
            map_recursive_zip(self.indirect_pass_grad, kwarg_values_2, kwargs)

    def replace_compressed_modules(self):
        """
        Replace all the modules that have changed (weights/inputs/output) shape.
        The new module is created using the same arguments of the to-be-replaced module,
        and correctly inherits its weights.

        NOTE: ```func``` type cannot be replaced as it is not a module, thus, one limitation
        is that ```func``` should be not required to be replaced.
        """
        # load the original stat dict before replace the model
        self.module.load_state_dict(self.ori_state_dict)
        self.logger.info("replace compressed modules...")
        # the mask conflict should be already resolved
        with torch.no_grad():
            for node in self.module.graph.nodes:
                self.replace_submodule(node)

    def replace_submodule(self, node: Node):
        """
        Replace the submodule according to the inferred sparsity.

        Parameters
        ----------
        unique_name: str
            The unique_name of the submodule to replace.
        reindex_dim: int
            The dimension of the re-index operation.
        reindex: Reindex
            The index tensor. Normally this variable is None. If we want to reindex the
            output of this submodule, we can pass the index by this parameter.
        """
        def tensors_flattener(masks):
            flattened = []
            def helper(obj):
                if isinstance(obj, torch.Tensor) and obj.numel() > 1:
                    flattened.append(obj)
            map_recursive(helper, masks)
            return flattened

        self.logger.debug("replace %s, with op_type %s", node.name, node.op)
        if node.op == 'call_module':
            node_info: NodeInfo = self.node_infos[node]
            sub_module: nn.Module = self.fetch_attr(node.target)
            sub_module_name = sub_module._get_name()

            if (not sub_module_name in replace_module) and (sub_module_name not in self.customized_replace_func):
                err_msg = f"Has not supported replacing module with type: {sub_module_name}, "
                err_msg += f"you could report an issue at https://github.com/microsoft/nni. "
                err_msg += f"If you know how to replace {sub_module_name}, "
                err_msg += f"you could implement module replacement by passing in"
                err_msg += f"`customized_replace_func` to `{self.__class__.__name__}`. "
                err_msg += f"You are welcome to contribute back to nni as native support if you have implemented the replacement function, "
                err_msg += f"so that more users can benefit from your contributions."
                raise RuntimeError(err_msg)
            self.logger.info("replace module (name: %s, op_type: %s)", node.name, sub_module_name)
            replace_function = self.customized_replace_func.get(sub_module_name, replace_module.get(sub_module_name, None))

            assert len(node.kwargs) == 0
            in_masks = tensors_flattener(map_recursive(self.slot_getter_mask_2_or_1, node.args))
            out_masks = map_recursive(self.slot_getter_mask_2_or_1, node)
            param_masks = node_info.param_masks_1

            compressed_module = replace_function(sub_module, (in_masks, out_masks, param_masks))

            new_submodule = compressed_module
            self.store_attr(node.target, compressed_module)
            return new_submodule
        else:
            return None

    def initialize_propagate(self, args):
        def model_tensor_randomizer(obj):
            if isinstance(obj, torch.Tensor) and obj.dim() > self.batch_dim:
                input_shape = list(obj.size())
                # set the batchsize to the confidence ratio
                input_shape[self.batch_dim] = self.batch_size
                return rand_like_with_shape(input_shape, obj)
            else:
                return obj

        self.logger.info('infer module masks...')

        self.ori_state_dict = copy.deepcopy(self.module.state_dict())

        # input of the whole model
        placeholder_names = [node.target for node in self.module.graph.nodes if node.op == 'placeholder']
        assert len(args) == len(placeholder_names)
        args = map_recursive(model_tensor_randomizer, args)
        self.arg_dict = {k: v for k, v in zip(placeholder_names, args)}

        # to store intermediate infomations
        self.slots: Dict[Node, Slot] = {node: Slot() for node in self.module.graph.nodes}

        # only for module now because only module can be replaced.
        self.node_infos: Dict[Node, NodeInfo] = {node: NodeInfo() for node in self.module.graph.nodes}
        for node in self.module.graph.nodes:
            node: Node
            for mask_updater in self.mask_updaters:
                if mask_updater.detect(self, node):
                    break
            # self.node_infos[node] = NodeInfo(mask_updater)
            # if node.op == 'call_module':
            #     sub_module: nn.Module = self.fetch_attr(node.target)
            #     param_masks = self.masks_file.get(node.target, {})
            #     for k, v in sub_module.named_parameters():
            #         if k not in param_masks:
            #             param_masks[k] = torch.ones_like(v)
            #     self.node_infos[node] = NodeInfo(param_masks)

    def run(self, *, args, masks_file, map_location=None) -> Any:
        """
        This class is to speedup the model with provided weight mask.

        Parameters
        ----------
        args : list
            the dummy_input to execute the model
        masks_file : str/dict
            The path of user provided mask file, or the mask object
        map_location : str
            the device on which masks are placed, same to map_location in ```torch.load```
        """

        if map_location is None:
            map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        if isinstance(masks_file, (str, Path)) and Path(masks_file).exists():
            self.masks_file = torch.load(masks_file, map_location)
        elif isinstance(masks_file, dict):
            self.masks_file = masks_file
        else:
            raise Exception('Please provide the mask or the path of the mask file')

        self.logger.info("start to speedup the model")
        training = self.module.training
        # set to the evaluation mode
        self.module.train(False)
        # TODO suppose to fix the conflict after the sparsity propagation
        # which is more elegent
        fix_mask_conflict(self.masks_file, self.module, args)

        self.initialize_propagate(args)

        self.propagate_originally()

        self.update_direct_sparsity()

        self.update_indirect_sparsity()

        self.logger.info('resolve the mask conflict')

        self.replace_compressed_modules()

        self.module.train(training)
        self.logger.info("speedup done")
