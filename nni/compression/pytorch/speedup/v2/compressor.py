# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy

import logging
from pathlib import Path
import queue
from typing import Any, Callable, Optional, Type
import functools

import torch
import torch.fx
from torch.fx import GraphModule
from torch.fx.node import Argument, Node, Target, map_aggregate
import torch.nn as nn

from nni.compression.pytorch.utils.mask_conflict import fix_mask_conflict
from nni.compression.pytorch.utils.utils import get_module_by_name, randomize_tensor, torch_float_dtype, torch_integer_dtype
from torch.fx._compatibility import compatibility
from nni.compression.pytorch.speedup.compress_modules import replace_module
from nni.common.concrete_trace_utils.utils import run_onlyif_instance, map_aggregate_zip


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

    def get_node_input(self, node: Node):
        pass
    def get_node_output(self, node: Node):
        pass
    def get_node_input_randomized(self, node: Node):
        pass

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
                randomize_tensor(new_obj, self.randomize_range_float[0], self.randomize_range_float[1])

        # apply input mask
        if obj in self.inter_var_mask:
            new_obj *= self.inter_var_mask[obj]
        
        return new_obj

    @run_onlyif_instance(Node)
    def arg_loader_detach(self, obj: Node) -> Any:
        if obj not in self.inter_vars:
            raise RuntimeError(f'Node {n} referenced nonexistent value {n_arg}! Run Graph.lint() '
                            f'to diagnose such issues')
        return map_aggregate(self.inter_vars[obj], self.tensor_detacher)

    @run_onlyif_instance(Node)
    def arg_loader_preprocess(self, obj: Node) -> Any:
        if obj not in self.inter_vars:
            raise RuntimeError(f'Node {n} referenced nonexistent value {n_arg}! Run Graph.lint() '
                            f'to diagnose such issues')
        return map_aggregate(self.inter_vars[obj], self.input_randomize_mask)

    @run_onlyif_instance(torch.Tensor, False)
    def get_in_mask(self, obj: torch.Tensor):
        if obj in self.inter_var_mask:
            return self.inter_var_mask[obj]
        return torch.ones_like(obj)

    def propagate_orig(self):
        for node in self.module.graph.nodes:
            node: Node
            
            self.logger.info('Propagate mask for %s', node.name)
            args = map_aggregate(node.args, self.arg_loader_detach)
            kwargs = map_aggregate(node.kwargs, self.arg_loader_detach)
            
            self.copied_args[node] = map_aggregate(args, self.tensor_cloner)
            self.copied_kwargs[node] = map_aggregate(kwargs, self.tensor_cloner)

            inter_var_orig = getattr(self, node.op)(node.target, args, kwargs)
            self.inter_vars[node] = inter_var_orig

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
                mean = torch.mean(obj, dim=0)
                mask_pos = std < self.STD_DELTA
                out_mask[:, mask_pos] = 0
            self.inter_var_mask[obj] = out_mask
            return out_mask
        return None
    
    def init_direct_mask(self, obj):
        if isinstance(obj, torch.Tensor):
            return torch.ones_like(obj)
        return None

    def has_special_handler_direct(self, node: Node) -> Optional[Callable]:
        return None

    def update_direct_sparsity(self):
        for node in self.module.graph.nodes:
            node: Node
            
            if node.op in ('placeholder', 'output'):
                continue

            self.logger.info('Update direct mask for %s', node.name)
            handler = self.has_special_handler_direct(node)
            if handler is not None:
                handler(self, node)
            elif node.op in ('call_function', 'call_method', 'call_module'):
                with torch.no_grad():
                    if node.op == 'call_module':
                        sub_module = self.fetch_attr(node.target)
                        weight_mask = {}
                        if node.target in self.masks_file:
                            weight_mask = self.masks_file[node.target]
                        for weight_key, weight_value in sub_module.named_parameters():
                            randomize_tensor(weight_value.data, self.randomize_range_float[0], self.randomize_range_float[1])
                            if weight_key not in weight_mask:
                                weight_mask[weight_key] = torch.ones_like(weight_value.data)
                            else:
                                weight_value.data *= weight_mask[weight_key]

                        self.weight_masks[node] = weight_mask

                    args = map_aggregate(node.args, self.arg_loader_preprocess)
                    kwargs = map_aggregate(node.kwargs, self.arg_loader_preprocess)
                    inter_var_masked = getattr(self, node.op)(node.target, args, kwargs)

                    args = map_aggregate(node.args, self.arg_loader_detach)
                    kwargs = map_aggregate(node.kwargs, self.arg_loader_detach)
                    self.args_masks[node] = map_aggregate(args, self.get_in_mask)
                    self.kwargs_masks[node] = map_aggregate(kwargs, self.get_in_mask)
                    self.out_masks[node] = map_aggregate(inter_var_masked, self.calc_direct_mask)
            else:
                args = map_aggregate(node.args, self.arg_loader_detach)
                kwargs = map_aggregate(node.kwargs, self.arg_loader_detach)
                self.args_masks[node] = map_aggregate(args, self.get_in_mask)
                self.kwargs_masks[node] = map_aggregate(kwargs, self.get_in_mask)
                self.out_masks[node] = map_aggregate(self.inter_vars[node], self.init_direct_mask)
            # do memory collect / compression

    def update_indirect_one_out_mask(self, obj_orig, obj_mask):
        if isinstance(obj_orig, torch.Tensor):
            assert isinstance(obj_mask, torch.Tensor)
            if obj_orig.grad is not None:
                gradient_sum = torch.sum(torch.abs(obj_orig.grad.data), dim=0)
                _grad_zero = gradient_sum == 0
                for batchid in range(obj_orig.size(0)):
                    # set the same mask value for the whole batche
                    obj_mask[batchid][_grad_zero] = 0

    def update_indirect_weight_mask(self, output, out_mask):
        # Note: output maybe tensor or list/tuple of tensors
        if isinstance(output, torch.Tensor):
            if output.grad_fn is not None:
                output.backward(out_mask)

    def update_indirect_weight_masks(self, node):
        args = map_aggregate(node.args, self.arg_loader_preprocess)
        kwargs = map_aggregate(node.kwargs, self.arg_loader_preprocess)
        output = getattr(self, node.op)(node.target, args, kwargs)
        if node.op == 'call_module':
            map_aggregate_zip(self.update_indirect_weight_mask, output, self.out_masks[node])
            
            # update the sparsity of the paramters
            sub_module = self.fetch_attr(node.target)
            for weight_key, weight_value in sub_module.named_parameters():
                grad_zero = weight_value.grad.data == 0
                self.weight_masks[node][weight_key][grad_zero] = 0
        else:
            map_aggregate_zip(self.update_indirect_weight_mask, output, self.out_masks[node])

    def requires_grad_(self, node, flag = True):
        """
        Set the requires_grad of input tensor and parameters to flag.
        """
        
        self.copied_args: Dict[Node, Any] = {}
        self.copied_kwargs: Dict[Node, Any] = {}
        if node in self.copied_args:
            def requires_grad_if_tensor(self, obj):
                if isinstance(obj, torch.Tensor):
                    # only float type can require the gradient
                    # enable the auto gradient
                    return obj.requires_grad_(flag)
                return None
            map_aggregate(self.copied_args[node], requires_grad_if_tensor)

        if node.op == 'call_module':
            sub_module = self.fetch_attr(node.target)
            for weight_key, weight_value in sub_module.named_parameters():
                if weight_value.dtype in torch_float_dtype:
                    weight_value.requires_grad_(flag)

    def update_indirect_sparsity(self):
        # for node in self.module.graph.nodes[::-1]:
        # for node in _node_list(self.module.graph):
        for node in reversed(self.module.graph.nodes):
            node: Node
            
            if node.op in ('placeholder', 'output'):
                continue
            
            self.logger.info('Update indirect mask for %s', node.name)
            out_orig = self.inter_vars[node]
            out_mask = self.out_masks[node]

            map_aggregate_zip(self.update_indirect_one_out_mask, out_orig, out_mask)

            self.requires_grad_(node)
            # Forward inference with auto gradient enabled
            # Note: tensors that need gradient cannot be used in the in-place operator
            # Some operator may have the in_place operations, so we need to clone the input
            # before passing to the self.module
            self.update_indirect_weight_masks(node)

            # # pass the gradient to the predecessor nodes
            # for in_id, tin in enumerate(auto_infer.dummy_input):
            #     debug_name = auto_infer.input_debugname[in_id]

            #     last_output = self.internal_result[debug_name]
            #     # if isinstance(last_output, torch.Tensor):
            #     # TODO what if last output is tuple/list of tensor
            #     if last_output.grad is not None and tin.grad is not None:
            #         last_output.grad.data += tin.grad.data
            #     elif last_output.grad is None:
            #         last_output.grad = tin.grad
            #     elif last_output.grad is not None and tin.grad is None:
            #         # for example, tin.view(batch, tin.size(1)/2, tin.view(2)*2)
            #         # the size operation of tin will have no gradient

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
                node: Node
                
                self.replace_submodule(node)

    def replace_submodule(self, node: Node, reindex_dim=None, reindex=None):
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
        class ReindexModule(nn.Module):
            """
            ReindexModule is used to resolve the mask conflict when replace the submodule.
            Basically, we can use two ways to resolve the mask conflict: (1) unmask some
            values(will introduce more computation overhead) (2) reindex and padd the output
            tensor of the target op(introduce more memory access overhad). Currently this
            method is shutdown, in the future, we will merge these two methods into a graph
            pass which is used to resolve the mask conflict.
            """

            def __init__(self, ori_module, reindex_dim, reindex):
                super(ReindexModule, self).__init__()
                self.ori_module = ori_module
                self.reindex_dim = reindex_dim
                self.reindex = reindex
                tmp_index = [slice(None, None) for i in range(reindex_dim+1)]
                # the index for the tensor
                tmp_index[reindex_dim] = reindex
                self.t_index = tuple(tmp_index)

            def forward(self, x):
                tmpout = self.ori_module(x)
                shape = list(tmpout.size())
                shape[self.reindex_dim] = self.reindex.size(0)
                out = torch.zeros(tuple(shape), device=tmpout.device,
                                  requires_grad=tmpout.requires_grad)
                out[self.t_index] = tmpout
                return out
        
        self.logger.debug("replace %s, with op_type %s", node.name, node.op)
        if node.op == 'call_module':
            # if g_node.unique_name in self.torch_graph.reused_module:
            #     if reindex_dim is not None:
            #         self.logger.warning(
            #             'Cannot replace a reused module with padding operator!!')
            #         return None
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
            masks = (self.args_masks[node], self.out_masks[node], self.weight_masks[node])
            compressed_module = replace_function(sub_module, masks)
            new_submodule = compressed_module
            if reindex_dim is None:
                self.store_attr(node.target, compressed_module)
            elif reindex_dim is not None and reindex is not None:
                # reindex the output of this submodule and replace the orginal module
                new_submodule = ReindexModule(compressed_module, reindex_dim, reindex)
                self.store_attr(node.target, compressed_module)
            return new_submodule
        else:
            return None

    def initialize_speedup(self, args):
        self.args_iter = iter(args)

        self.ori_state_dict = copy.deepcopy(self.module.state_dict())
        self.inter_vars: Dict[Node, Any] = {}

        self.copied_args: Dict[Node, Any] = {}
        self.copied_kwargs: Dict[Node, Any] = {}
        # self.copied_output: Dict[Node, Any] = {}

        self.inter_var_mask: Dict[torch.Tensor, torch.Tensor] = {}
        self.args_masks: Dict[Node, Any] = {}
        self.kwargs_masks: Dict[Node, Any] = {}
        self.out_masks: Dict[Node, Any] = {}
        self.weight_masks: Dict[Node, Any] = {}

    def run(self, *args) -> Any:
        """
        There are basically two steps: first, do mask/shape inference,
        second, replace modules.
        """

        self.logger.info("start to speedup the model")
        self.initialize_speedup(args)
        training = self.module.training
        # set to the evaluation mode
        self.module.train(False)
        # TODO suppose to fix the conflict after the sparsity propagation
        # which is more elegent
        fix_mask_conflict(self.masks_file, self.module, args)

        self.logger.info('propagate forward')
        self.propagate_orig()
        self.logger.info('propagate forward end')
        self.logger.info('infer module masks...')
        self.update_direct_sparsity()
        self.update_indirect_sparsity()
        self.logger.info('resolve the mask conflict')

        # load the original stat dict before replace the model
        self.module.load_state_dict(self.ori_state_dict)
        self.logger.info("replace compressed modules...")
        # the mask conflict should be already resolved
        self.replace_compressed_modules()
        self.module.train(training)
        self.logger.info("speedup done")
