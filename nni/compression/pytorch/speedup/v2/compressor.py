# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy

import logging
from pathlib import Path
import queue
from typing import Any, Callable, Optional, Type, Dict, Union
import functools

import torch
import torch.fx
from torch.fx import GraphModule
from torch.fx.node import Argument, Node, Target
import torch.nn as nn

from nni.compression.pytorch.utils.mask_conflict import fix_mask_conflict
from nni.compression.pytorch.utils.utils import get_module_by_name, randomize_tensor, torch_float_dtype, torch_integer_dtype
from torch.fx._compatibility import compatibility
from nni.compression.pytorch.speedup.compress_modules import replace_module
from nni.common.concrete_trace_utils.utils import run_onlyif_instance, map_recursive, map_recursive_zip
from nni.compression.pytorch.speedup.v2.container import Slot, NodeInfo


@compatibility(is_backward_compatible=True)
class ModelSpeedup(torch.fx.Interpreter):
    STD_DELTA = 1e-6
    randomize_range_float = (0.1, 8.0)

    def __init__(self,
                module: GraphModule,
                masks_file,
                map_location=None,
                batch_dim=0,
                batch_size=8,
                customized_replace_func = dict(),
                garbage_collect_values: bool = True,
                logger:Optional[logging.Logger] = None):
        super().__init__(module, garbage_collect_values)

        self.module: GraphModule
        self.masks_file = masks_file
        self.map_location = map_location
        self.batch_dim = batch_dim
        self.batch_size = batch_size
        self.customized_replace_func = customized_replace_func
        if logger == None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
        else:
            self.logger = logger

    def has_special_handler_indirect(self, node: Node) -> Optional[Callable]:
        pass
    
    @compatibility(is_backward_compatible=True)
    def store_attr(self, path: str, obj):
        target_atoms = path.split('.')
        attr_itr = self.module
        # for i, atom in enumerate(target_atoms)[:-1]:
        #     if not hasattr(attr_itr, atom):
        #         raise RuntimeError(f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}")
        #     attr_itr = getattr(attr_itr, atom)
        # setattr(attr_itr, obj)
        for i in range(len(target_atoms))[:-1]:
            atom = target_atoms[i]
            if not hasattr(attr_itr, atom):
                raise RuntimeError(f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}")
            attr_itr = getattr(attr_itr, atom)
        setattr(attr_itr, target_atoms[-1], obj)

    @run_onlyif_instance(torch.Tensor)
    def tensor_detacher(self, obj: torch.Tensor):
        return obj.detach()

    @run_onlyif_instance(torch.Tensor)
    def tensor_cloner(self, obj: torch.Tensor):
        return obj.clone()

    @run_onlyif_instance(torch.Tensor)
    def input_randomize_mask(self, obj: torch.Tensor):
        # detach
        new_obj = obj.detach()

        # randomize
        if len(new_obj.size()) > self.batch_dim and new_obj.size(self.batch_dim) == self.batch_size:
            with torch.no_grad():
                self.seed_inter_var += 1
                torch.manual_seed(self.seed_inter_var)
                randomize_tensor(new_obj.data, self.randomize_range_float[0], self.randomize_range_float[1])
                torch.manual_seed(100)

        # apply input mask
        if id(obj) in self.inter_var_mask:
            # change data with no-in-place operators, so the data_ptr() changed
            new_obj.data = new_obj.data * self.inter_var_mask[id(obj)]
            # new_obj.data *= self.inter_var_mask[id(obj)] # in-place and data_ptr() not changed
            # new_obj *= self.inter_var_mask[id(obj)]
        
        return new_obj

    @run_onlyif_instance(Node)
    def arg_loader_orig(self, obj: Node) -> Any:
        if obj not in self.inter_vars:
            raise RuntimeError(f'Node {obj} referenced nonexistent value {obj.args}! Run Graph.lint() '
                            f'to diagnose such issues')
        return self.inter_vars[obj]

    @run_onlyif_instance(Node)
    def arg_loader_detach(self, obj: Node) -> Any:
        if obj not in self.inter_vars:
            raise RuntimeError(f'Node {obj} referenced nonexistent value {obj.args}! Run Graph.lint() '
                            f'to diagnose such issues')
        return map_recursive(self.tensor_detacher, self.inter_vars[obj])

    @run_onlyif_instance(Node)
    def arg_loader_preprocess(self, obj: Node) -> Any:
        if obj not in self.inter_vars:
            raise RuntimeError(f'Node {obj} referenced nonexistent value {obj.args}! Run Graph.lint() '
                            f'to diagnose such issues')
        return map_recursive(self.input_randomize_mask, self.inter_vars[obj])

    @run_onlyif_instance(torch.Tensor, False)
    def get_in_mask(self, obj: torch.Tensor):
        if id(obj) in self.inter_var_mask:
            return self.inter_var_mask[id(obj)]
        return torch.ones_like(obj)

    def debug_hash_value_one(self, obj):
        if isinstance(obj, torch.Tensor) and obj.numel() > 1:
            torch.manual_seed(100)
            out = torch.sum(torch.rand_like(obj.to(torch.float)) * obj).item()
            if obj.grad is not None:
                torch.manual_seed(100)
                grad = torch.sum(torch.rand_like(obj.grad.to(torch.float)) * obj.grad).item()
                return [out, 'grad: %s' % grad]
            else:
                return [out, 'no grad']
        else:
            return obj

    def debug_hash_value(self, obj):
        return map_recursive(self.debug_hash_value_one, obj)

    def tensor_propagate_check(self, obj: torch.Tensor):
        # return isinstance(obj, torch.Tensor) and obj.dim() > self.batch_dim and obj.size(self.batch_dim) == self.confidence
        return obj.numel() > self.confidence and obj.numel() % self.confidence == 0

    def tensor_detacher(self, obj: torch.Tensor):
        if isinstance(obj, torch.Tensor):
            return obj.detach()
        return obj

    def tensor_cloner(self, obj: torch.Tensor):
        if isinstance(obj, torch.Tensor):
            return obj.clone()
        return obj

    def tensor_clone_detacher(self, obj: torch.Tensor):
        if isinstance(obj, torch.Tensor):
            return obj.clone().detach()
        return obj

    def slot_getter_value_0(self, node: Node):
        assert isinstance(node, Node)
        assert self.slots[node].status['value_0'] == 1, 'slot error: bad value_0(%d)' % self.slots[node].status['value_0']

        return self.slots[node].value_0

    def slot_getter_value_1(self, node: Node):
        assert isinstance(node, Node)
        assert self.slots[node].status['value_1'] == 1, 'slot error: bad value_1(%d)' % self.slots[node].status['value_1']

        return self.slots[node].value_1

    def slot_getter_value_2(self, node: Node):
        assert isinstance(node, Node)
        assert self.slots[node].status['value_2'] >= 1, 'slot error: bad value_2(%d)' % self.slots[node].status['value_2']

        return self.slots[node].value_2

    def clone_randomizer(self, obj):
        import copy
        # if self.tensor_propagate_check(obj):
        if isinstance(obj, torch.Tensor)
            if obj.numel() != 1 and len(obj.size()) > self.batch_dim and obj.size(self.batch_dim) == self.confidence:
                new_obj = obj.clone().detach()
                if not new_obj.is_contiguous():
                    new_obj = new_obj.contiguous()
                randomize_tensor(new_obj, start=0.1, end=8.0)
                return new_obj
            else:
                new_obj = obj.clone().detach()
                return new_obj
        else:
            try:
                return copy.deepcopy(obj)
            except copy.Error:
                return obj

    def slot_getter_mask_1(self, node: Node):
        assert isinstance(node, Node)
        assert self.slots[node].status['mask_1'] == 1, 'slot error: bad mask_1(%d)' % self.slots[node].status['mask_1']

        return self.slots[node].mask_1

    def slot_getter_mask_2(self, node: Node):
        assert isinstance(node, Node)
        if self.slots[node].mask_2 is None:
            return None
        else:
            assert self.slots[node].status['mask_2'] >= 1, 'slot error: bad mask_2(%d)' % self.slots[node].status['mask_2']

            return self.slots[node].mask_2

    def slot_getter_mask_2_or_1(self, node: Node):
        assert isinstance(node, Node)
        if self.slots[node].mask_2 is None:
            assert self.slots[node].status['mask_1'] == 1, 'slot error: bad mask_1(%d)' % self.slots[node].status['mask_1']

            return self.slots[node].mask_1
        else:
            assert self.slots[node].status['mask_2'] >= 1, 'slot error: bad mask_2(%d)' % self.slots[node].status['mask_2']

            return self.slots[node].mask_2

    def mask_applier(self, value, mask):
        if self.tensor_propagate_check(value):
            assert isinstance(mask, torch.Tensor) and value.shape == mask.shape
            return value * mask
        else:
            assert mask is None
            return value

    def calc_one_mask(self, obj):
        if self.tensor_propagate_check(obj):
            obj: torch.Tensor
            STD_DELTA = 1e-6

            out_mask = torch.ones_like(obj)
            if obj.dtype in torch_integer_dtype:
                same = obj[:] == obj[0]
                reduced = torch.sum(same, dim=0)
                is_constant = reduced == obj.size(0)
                out_mask[:, is_constant] = 0
            else:
                std = torch.std(obj, dim=0)
                mask_pos = std < STD_DELTA
                out_mask[:, mask_pos] = 0
            return out_mask
        else:
            return None

    def tensor_requires_grad(self, obj):
        if self.tensor_propagate_check(obj) and obj.dtype in torch_float_dtype:
            # only float type can require the gradient
            # enable the auto gradient
            obj.requires_grad_(True)

    # @compatibility(is_backward_compatible=True)
    # def placeholder(self, target, args, kwargs) -> Any:
        
    #     @run_onlyif_instance(torch.Tensor)
    #     def input_init(obj: torch.Tensor):
    #         return torch.ones_like(obj)

    #     assert isinstance(target, str)
    #     if target.startswith('*'):
    #         return list(self.args_iter)
    #     else:
    #         try:
    #             return map_recursive(input_init, next(self.args_iter))
    #         except StopIteration as si:
    #             if len(args) > 0:
    #                 return args[0]
    #             else:
    #                 raise RuntimeError(f'Expected positional argument for parameter {target}, but one was not passed in!')

    def propagate_orig(self):
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

            # do memory collect / compression
            if self.garbage_collect_values:
                # for to_delete in self.user_to_last_uses.get(node, []):
                #     del self.inter_vars[to_delete]
                pass

    def calc_direct_mask(self, obj):
        if isinstance(obj, torch.Tensor):
            out_mask = torch.ones_like(obj)
            # judge if tout is a scalar(tensor that only have one value)
            if len(obj.size()) == 0:
                # obj is a scalar tensor, for the scalar tensor, we take
                # this scalar as a constant, usually, the scalar tensor is returned
                # by the size() function
                # return out_mask
                pass
            elif obj.dtype in torch_integer_dtype:
                # Pytorch cannot use torch.mean and torch.std to process
                # intergers :( , so if dtype of the input tensor is integer, we need
                # check if is the constant by ourselves
                # Note: the first dimension should be the batch dimension
                same = obj[:] == obj[0]
                reduced = torch.sum(same, dim=0)
                is_constant = reduced == obj.size(0)
                out_mask[:, is_constant] = 0
            else:
                # calculate the std of the output among batch dimension
                std = torch.std(obj, dim=0)
                # calculate the mean value of the output among the batch dimension
                mask_pos = std < self.STD_DELTA
                out_mask[:, mask_pos] = 0
            self.inter_var_mask[id(obj)] = out_mask
            return out_mask
        return None
    
    def calc_direct_mask2(self, obj_in_orig, obj_proced):
        if isinstance(obj_in_orig, torch.Tensor):
            assert isinstance(obj_proced, torch.Tensor)
            assert obj_in_orig.shape == obj_proced.shape

            out_mask = torch.ones_like(obj_proced)
            # judge if tout is a scalar(tensor that only have one value)
            if len(obj_proced.size()) == 0:
                # obj is a scalar tensor, for the scalar tensor, we take
                # this scalar as a constant, usually, the scalar tensor is returned
                # by the size() function
                # return out_mask
                pass
            elif obj_proced.dtype in torch_integer_dtype:
                # Pytorch cannot use torch.mean and torch.std to process
                # intergers :( , so if dtype of the input tensor is integer, we need
                # check if is the constant by ourselves
                # Note: the first dimension should be the batch dimension
                same = obj_proced[:] == obj_proced[0]
                reduced = torch.sum(same, dim=0)
                is_constant = reduced == obj_proced.size(0)
                out_mask[:, is_constant] = 0
            else:
                # calculate the std of the output among batch dimension
                std = torch.std(obj_proced, dim=0)
                # calculate the mean value of the output among the batch dimension
                mask_pos = std < self.STD_DELTA
                out_mask[:, mask_pos] = 0
            self.inter_var_mask[id(obj_in_orig)] = out_mask
            return out_mask
        return None

    def init_direct_mask(self, obj):
        if isinstance(obj, torch.Tensor):
            return torch.ones_like(obj)
        return None

    def has_special_handler_direct(self, node: Node) -> Optional[Callable]:
        return None

    def update_direct_sparsity(self):
        # update indirect out mask

        self.logger.info("update direct sparsity")

        for node in self.module.graph.nodes:
            node: Node

            self.slots[node].value_2 = map_recursive(self.clone_randomizer, self.slots[node].value_0)
            self.slots[node].status['value_2'] += 1

        for node in self.module.graph.nodes:
            node: Node

            if node.op in ('placeholder', 'output'):
                continue

            self.logger.info('Update direct mask for %s: %s', node.op, node.name)
            handler = self.has_special_handler_direct(node)
            if handler is not None:
                handler(self, node)
            elif node.op in ('call_function', 'call_method', 'call_module'):
                with torch.no_grad():
                    if node.op == 'call_module':
                        node_info: NodeInfo = self.node_infos[node]
                        sub_module: nn.Module = self.fetch_attr(node.target)

                        for _k, v in sub_module.named_parameters():
                            randomize_tensor(v.data, self.randomize_range_float[0], self.randomize_range_float[1])

                        for k, v in sub_module.named_parameters():
                            v *= node_info.param_masks_0[k] # in-place addition
                            # sub_module.register_parameter(
                            #     k,
                            #     torch.nn.Parameter(v * node_info.param_masks_0[k])
                            # )

                    args = map_recursive(self.slot_getter_value_2, node.args)
                    arg_masks = map_recursive(self.slot_getter_mask_1, node.args)
                    args = map_recursive_zip(self.mask_applier, args, arg_masks)
                    kwargs = map_recursive(self.slot_getter_value_2, node.kwargs)
                    kwarg_masks = map_recursive(self.slot_getter_mask_1, node.kwargs)
                    kwargs = map_recursive_zip(self.mask_applier, kwargs, kwarg_masks)

                    output = getattr(self, node.op)(node.target, args, kwargs)

                    self.slots[node].mask_1 = map_recursive(self.calc_one_mask, output)
                    self.slots[node].status['mask_1'] += 1

            # do memory collect / compression

    def update_indirect_one_out_mask(self, obj_orig, obj_mask):
        if isinstance(obj_orig, torch.Tensor):
            assert isinstance(obj_mask, torch.Tensor)
            if obj_orig.grad is not None:
                # todo: grad is bad
                gradient_sum = torch.sum(torch.abs(obj_orig.grad.data), dim=0)
                _grad_zero = gradient_sum == 0
                for batchid in range(obj_orig.size(0)):
                    # set the same mask value for the whole batche
                    obj_mask[batchid][_grad_zero] = 0

    def update_indirect_weight_mask(self, output, out_mask):
        # Note: output maybe tensor or list/tuple of tensors
        if isinstance(output, torch.Tensor):
            assert isinstance(out_mask, torch.Tensor)
            if output.grad_fn is not None:
                output.backward(out_mask)
        else:
            assert not isinstance(out_mask, torch.Tensor)

    def update_indirect_sparsity(self):
        # update indirect out mask
        def calc_indirect_mask(mask, obj):
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

        def update_indirect_weight_mask_helper(output, mask):
            # Note: output maybe tensor or list/tuple of tensors
            if isinstance(output, torch.Tensor) and self.tensor_propagate_check(output):
                assert isinstance(mask, torch.Tensor)
                if output.grad_fn is not None:
                    output.backward(mask)
            else:
                assert not isinstance(mask, torch.Tensor)

        # # pass the gradient to the predecessor nodes
        def pass_grad(slot_val, out):
            # Note: output maybe tensor or list/tuple of tensors
            if isinstance(slot_val, torch.Tensor)
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

        self.logger.info("update indirect sparsity")

        for node in reversed(self.module.graph.nodes):
            node: Node
            
            if node.op in ('placeholder', 'output'):
                continue

            self.logger.info('Update indirect mask for %s: %s', node.op, node.name)

            # print('inter var before indirect propagation:', node.name)
            # print('in_masks:', [(torch.manual_seed(100), torch.sum(torch.rand_like(i.to(torch.float)) * i))[1] if isinstance(i, torch.Tensor) else i for i in self.args_masks[node]])
            # print('out_masks:', (torch.manual_seed(100), torch.sum(torch.rand_like(self.out_masks[node].to(torch.float)) * self.out_masks[node]))[1] if isinstance(self.out_masks[node], torch.Tensor) else self.out_masks[node])
            # # print('dummy_input:', [(torch.manual_seed(100), torch.sum(torch.rand_like(i.to(torch.float)) * i))[1] for i in auto_infer.dummy_input])
            # print('orig_output:', (torch.manual_seed(100), torch.sum(torch.rand_like(self.inter_vars[node].to(torch.float)) * self.inter_vars[node]))[1] if isinstance(self.inter_vars[node], torch.Tensor) else self.inter_vars[node])
            # print('orig_output.grad is None:', self.inter_vars[node].grad is None if isinstance(self.inter_vars[node], torch.Tensor) else self.inter_vars[node])
            # print('seeds:', self.seed_inter_var, self.seed_weight)

            output = map_recursive(self.slot_getter_value_2, node)
            output_masks_1 = map_recursive(self.slot_getter_mask_1, node)
            output_masks_2 = map_recursive_zip(calc_indirect_mask, output_masks_1, output)

            self.slots[node].mask_2 = output_masks_2
            self.slots[node].status['mask_2'] += 1

            # init apply input
            # randomized, so it's same to use slot_getter_value_orig or slot_getter_value_orig_inplace
            args = map_recursive(self.slot_getter_value_0, node.args)
            args = map_recursive(self.clone_randomizer, args)
            arg_masks = map_recursive(self.slot_getter_mask_1, node.args)
            args = map_recursive_zip(self.mask_applier, args, arg_masks)
            map_recursive(self.tensor_requires_grad, args)

            kwargs = map_recursive(self.slot_getter_value_0, node.kwargs)
            kwargs = map_recursive(self.clone_randomizer, kwargs)
            kwarg_masks = map_recursive(self.slot_getter_mask_1, node.kwargs)
            kwargs = map_recursive_zip(self.mask_applier, kwargs, kwarg_masks)
            map_recursive(self.tensor_requires_grad, kwargs)

            output = getattr(self, node.op)(node.target, args, kwargs)
            
            map_recursive_zip(update_indirect_weight_mask_helper, output, output_masks_2)

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

            map_recursive_zip(pass_grad, arg_values_2, args)
            map_recursive_zip(pass_grad, kwarg_values_2, kwargs)

            # print('inter var after indirect propagation:', node.name)
            # print('in_masks:', [(torch.manual_seed(100), torch.sum(torch.rand_like(i.to(torch.float)) * i))[1] if isinstance(i, torch.Tensor) else i for i in self.args_masks[node]])
            # print('out_masks:', (torch.manual_seed(100), torch.sum(torch.rand_like(self.out_masks[node].to(torch.float)) * self.out_masks[node]))[1] if isinstance(self.out_masks[node], torch.Tensor) else self.out_masks[node])
            # # print('dummy_input:', [(torch.manual_seed(100), torch.sum(torch.rand_like(i.to(torch.float)) * i))[1] for i in auto_infer.dummy_input])
            # print('orig_output:', (torch.manual_seed(100), torch.sum(torch.rand_like(self.inter_vars[node].to(torch.float)) * self.inter_vars[node]))[1] if isinstance(self.inter_vars[node], torch.Tensor) else self.inter_vars[node])
            # print('orig_output.grad is None:', self.inter_vars[node].grad is None if isinstance(self.inter_vars[node], torch.Tensor) else self.inter_vars[node])
            # print('seeds:', self.seed_inter_var, self.seed_weight)

    def replace_compressed_modules(self):
        """
        Replace all the modules that have changed (weights/inputs/output) shape.
        The new module is created using the same arguments of the to-be-replaced module,
        and correctly inherits its weights.

        NOTE: ```func``` type cannot be replaced as it is not a module, thus, one limitation
        is that ```func``` should be not required to be replaced.
        """
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
            
            # print('inter var in replacement:')
            # print('replace module:', node.name)
            # print('replace in_masks:')
            # print([(torch.manual_seed(100), torch.sum(torch.rand_like(v.to(torch.float)) * v))[1] for v in self.args_masks[node]])
            # print('replace out_masks:')
            # print([(torch.manual_seed(100), torch.sum(torch.rand_like(self.out_masks[node].to(torch.float)) * self.out_masks[node]))[1]])
            # print('replace weight_mask:')
            # print({k: (torch.manual_seed(100), torch.sum(torch.rand_like(v.to(torch.float)) * v))[1] for k, v in self.weight_masks[node].items()})

            new_submodule = compressed_module
            self.store_attr(node.target, compressed_module)
            return new_submodule
        else:
            return None

    def initialize_speedup(self, args):
        self.logger.info('infer module masks...')

        self.ori_state_dict = copy.deepcopy(self.module.state_dict())

        # input of the whole model
        self.args_iter = iter(args)

        # to store intermediate infomations
        self.slots: Dict[Node, Slot] = {node: Slot() for node in self.module.graph.nodes}

        # only for module now because only module can be replaced.
        self.node_infos: Dict[Node, NodeInfo] = {}
        for node in self.module.graph.nodes:
            node: Node
            if node.op == 'call_module':
                sub_module: nn.Module = self.fetch_attr(node.target)
                param_masks = self.masks_file.get(node.name, {})
                for k, v in sub_module.named_parameters():
                    if k not in param_masks:
                        param_masks[k] = torch.ones_like(v)
                self.node_infos[node] = NodeInfo(
                    param_masks
                )

    def run(self, *args) -> Any:
        """
        There are basically two steps: first, do mask/shape inference,
        second, replace modules.
        """

        self.logger.info("start to speedup the model")
        training = self.module.training
        # set to the evaluation mode
        self.module.train(False)
        # TODO suppose to fix the conflict after the sparsity propagation
        # which is more elegent
        fix_mask_conflict(self.masks_file, self.module, args)

        self.initialize_speedup(args)

        # torch.manual_seed(100)
        # self.seed_inter_var = 1000
        # self.seed_weight = 1000
        self.propagate_orig()
        # print('inter var orig_output1:')
        # print([(k, torch.sum(v) if isinstance(v, torch.Tensor) else v) for k, v in self.inter_vars.items()])
        # print('inter var orig_output.grad1:')
        # print([(k, v.grad is None if isinstance(v, torch.Tensor) else v) for k, v in self.inter_vars.items()])

        # torch.manual_seed(100)
        # self.seed_inter_var = 1000
        # self.seed_weight = 1000
        self.update_direct_sparsity()
        # print('inter var2:')
        # print([(k, [(torch.manual_seed(100), torch.sum(torch.rand_like(i.to(torch.float)) * i))[1] if isinstance(i, torch.Tensor) else i for i in v]) for k, v in self.args_masks.items()])
        # print('inter var3:')
        # print([(k, (torch.manual_seed(100), torch.sum(torch.rand_like(v.to(torch.float)) * v))[1] if isinstance(v, torch.Tensor) else v) for k, v in self.out_masks.items()])
        # print('inter var4:')
        # print([(ko, [(ki, (torch.manual_seed(100), torch.sum(torch.rand_like(vi.to(torch.float)) * vi))[1]) for ki, vi in vo.items()]) for ko, vo in self.weight_masks.items() if ko.op == 'call_module'])
        # print('inter var4.5:')
        # print([(k, v.grad is None if isinstance(v, torch.Tensor) else v) for k, v in self.inter_vars.items()])
        # print('inter var orig_output2:')
        # print([(k, torch.sum(v) if isinstance(v, torch.Tensor) else v) for k, v in self.inter_vars.items()])

        # torch.manual_seed(100)
        # self.seed_inter_var = 1000
        # self.seed_weight = 1000
        self.update_indirect_sparsity()
        # print('inter var5:')
        # print([(k, (torch.manual_seed(100), torch.sum(torch.rand_like(v.to(torch.float)) * v))[1] if isinstance(v, torch.Tensor) else v) for k, v in self.out_masks.items()])
        self.logger.info('resolve the mask conflict')

        # load the original stat dict before replace the model
        self.module.load_state_dict(self.ori_state_dict)
        self.logger.info("replace compressed modules...")
        # the mask conflict should be already resolved
        self.replace_compressed_modules()
        self.module.train(training)
        self.logger.info("speedup done")
