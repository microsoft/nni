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
        for node in self.module.graph.nodes:
            node: Node
            
            if node.op in ('call_function', 'call_method'):
                self.logger.info('Propagate variables for call: %s', node.name)
            elif node.op in ('call_module'):
                self.logger.info('Propagate variables for module: %s', node.name)
            else:
                self.logger.info('Propagate variables for %s: %s', node.op, node.name)

            args = map_recursive(self.arg_loader_detach, node.args)
            kwargs = map_recursive(self.arg_loader_detach, node.kwargs)
            
            self.copied_args[node] = map_recursive(self.tensor_cloner, args)
            self.copied_kwargs[node] = map_recursive(self.tensor_cloner, kwargs)

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
        for node in self.module.graph.nodes:
            node: Node

            if node.op in ('placeholder', 'output'):
                continue

            if node.op in ('call_function', 'call_method'):
                self.logger.info('Update direct mask for call: %s', node.name)
            elif node.op in ('call_module'):
                self.logger.info('Update direct mask for module: %s', node.name)
            else:
                assert False
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
                            self.seed_weight += 1
                            torch.manual_seed(self.seed_weight)
                            randomize_tensor(weight_value.data, self.randomize_range_float[0], self.randomize_range_float[1])
                            torch.manual_seed(100)
                            if weight_key not in weight_mask:
                                weight_mask[weight_key] = torch.ones_like(weight_value.data)
                            else:
                                weight_value.data *= weight_mask[weight_key]

                        self.weight_masks[node] = weight_mask

                    proced_args = map_recursive(self.arg_loader_preprocess, node.args)
                    proced_kwargs = map_recursive(self.arg_loader_preprocess, node.kwargs)
                    proced_out = getattr(self, node.op)(node.target, proced_args, proced_kwargs)

                    self.args_masks[node] = map_recursive(self.get_in_mask, map_recursive(self.arg_loader_detach, node.args))
                    self.kwargs_masks[node] = map_recursive(self.get_in_mask, map_recursive(self.arg_loader_detach, node.kwargs))
                    # self.out_masks[node] = map_recursive(self.calc_direct_mask, proced_out)
                    self.out_masks[node] = map_recursive_zip(self.calc_direct_mask2, self.inter_vars[node], proced_out)
            else:
                self.args_masks[node] = map_recursive(self.get_in_mask, map_recursive(self.arg_loader_detach, node.args))
                self.kwargs_masks[node] = map_recursive(self.get_in_mask, map_recursive(self.arg_loader_detach, node.kwargs))
                self.out_masks[node] = map_recursive(self.init_direct_mask, self.inter_vars[node])
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

    def requires_grad_(self, node, args, kwargs, flag = True):
        """
        Set the requires_grad of input tensor and parameters to flag.
        """
        @run_onlyif_instance(torch.Tensor, False)
        def requires_grad_if_tensor(obj: torch.Tensor):
            # only float type can require the gradient
            # enable the auto gradient
            if obj.dtype in torch_float_dtype:
                obj.requires_grad_(flag)

        map_recursive(requires_grad_if_tensor, args)
        map_recursive(requires_grad_if_tensor, kwargs)

        if node.op == 'call_module':
            sub_module = self.fetch_attr(node.target)
            for weight_key, weight_value in sub_module.named_parameters():
                if weight_value.dtype in torch_float_dtype:
                    weight_value.requires_grad_(flag)

    def update_indirect_sparsity(self):
        for node in reversed(self.module.graph.nodes):
            node: Node
            
            if node.op in ('placeholder', 'output'):
                continue
            
            if node.op in ('call_function', 'call_method'):
                self.logger.info('Update indirect mask for call: %s', node.name)
            elif node.op in ('call_module'):
                self.logger.info('Update indirect mask for module: %s', node.name)
            else:
                assert False

            print('inter var before indirect propagation:', node.name)
            print('in_masks:', [(torch.manual_seed(100), torch.sum(torch.rand_like(i.to(torch.float)) * i))[1] if isinstance(i, torch.Tensor) else i for i in self.args_masks[node]])
            print('out_masks:', (torch.manual_seed(100), torch.sum(torch.rand_like(self.out_masks[node].to(torch.float)) * self.out_masks[node]))[1] if isinstance(self.out_masks[node], torch.Tensor) else self.out_masks[node])
            # print('dummy_input:', [(torch.manual_seed(100), torch.sum(torch.rand_like(i.to(torch.float)) * i))[1] for i in auto_infer.dummy_input])
            print('orig_output:', (torch.manual_seed(100), torch.sum(torch.rand_like(self.inter_vars[node].to(torch.float)) * self.inter_vars[node]))[1] if isinstance(self.inter_vars[node], torch.Tensor) else self.inter_vars[node])
            print('orig_output.grad is None:', self.inter_vars[node].grad is None if isinstance(self.inter_vars[node], torch.Tensor) else self.inter_vars[node])
            print('seeds:', self.seed_inter_var, self.seed_weight)

            out_orig = self.inter_vars[node]
            out_mask = self.out_masks[node]

            map_recursive_zip(self.update_indirect_one_out_mask, out_orig, out_mask)

            # Forward inference with auto gradient enabled
            # Note: tensors that need gradient cannot be used in the in-place operator
            # Some operator may have the in_place operations, so we need to clone the input
            # before passing to the self.module
            proced_args = map_recursive(self.arg_loader_preprocess, node.args)
            proced_kwargs = map_recursive(self.arg_loader_preprocess, node.kwargs)
            
            if node.op == 'call_module':
                sub_module = self.fetch_attr(node.target)
                weight_mask = {}
                if node.target in self.masks_file:
                    weight_mask = self.masks_file[node.target]
                for weight_key, weight_value in sub_module.named_parameters():
                    self.seed_weight += 1
                    torch.manual_seed(self.seed_weight)
                    randomize_tensor(weight_value.data, self.randomize_range_float[0], self.randomize_range_float[1])
                    torch.manual_seed(100)
                    if weight_key not in weight_mask:
                        weight_mask[weight_key] = torch.ones_like(weight_value.data)
                    else:
                        weight_value.data *= weight_mask[weight_key]

                self.weight_masks[node] = weight_mask

            self.requires_grad_(node, proced_args, proced_kwargs) # ??
            # Some operator may have the in_place operations, so we need to clone the input
            # before passing to the self.module
            cloned_proced_args = map_recursive(self.tensor_cloner, proced_args)
            cloned_proced_kwargs = map_recursive(self.tensor_cloner, proced_kwargs)
            output = getattr(self, node.op)(node.target, cloned_proced_args, cloned_proced_kwargs)

            map_recursive_zip(self.update_indirect_weight_mask, output, self.out_masks[node])

            if node.op == 'call_module':
                # update the sparsity of the paramters
                sub_module = self.fetch_attr(node.target)
                for weight_key, weight_value in sub_module.named_parameters():
                    grad_zero = weight_value.grad.data == 0
                    self.weight_masks[node][weight_key][grad_zero] = 0

            # # pass the gradient to the predecessor nodes
            def pass_gradient(in_orig, in_propagated):
                # Note: output maybe tensor or list/tuple of tensors
                if isinstance(in_orig, torch.Tensor):
                    assert isinstance(in_propagated, torch.Tensor)
                    if in_orig.grad is not None and in_propagated.grad is not None:
                        in_orig.grad.data += in_propagated.grad.data
                    elif in_orig.grad is None:
                        in_orig.grad = in_propagated.grad
                    elif in_orig.grad is not None and in_propagated.grad is None:
                        # for example, tin.view(batch, tin.size(1)/2, tin.view(2)*2)
                        # the size operation of tin will have no gradient
                        pass
                else:
                    assert not isinstance(in_propagated, torch.Tensor)
            orig_args = map_recursive(self.arg_loader_orig, node.args)
            orig_kwargs = map_recursive(self.arg_loader_orig, node.kwargs)

            map_recursive_zip(pass_gradient, orig_args, proced_args)
            map_recursive_zip(pass_gradient, orig_kwargs, proced_kwargs)
            
            print('inter var after indirect propagation:', node.name)
            print('in_masks:', [(torch.manual_seed(100), torch.sum(torch.rand_like(i.to(torch.float)) * i))[1] if isinstance(i, torch.Tensor) else i for i in self.args_masks[node]])
            print('out_masks:', (torch.manual_seed(100), torch.sum(torch.rand_like(self.out_masks[node].to(torch.float)) * self.out_masks[node]))[1] if isinstance(self.out_masks[node], torch.Tensor) else self.out_masks[node])
            # print('dummy_input:', [(torch.manual_seed(100), torch.sum(torch.rand_like(i.to(torch.float)) * i))[1] for i in auto_infer.dummy_input])
            print('orig_output:', (torch.manual_seed(100), torch.sum(torch.rand_like(self.inter_vars[node].to(torch.float)) * self.inter_vars[node]))[1] if isinstance(self.inter_vars[node], torch.Tensor) else self.inter_vars[node])
            print('orig_output.grad is None:', self.inter_vars[node].grad is None if isinstance(self.inter_vars[node], torch.Tensor) else self.inter_vars[node])
            print('seeds:', self.seed_inter_var, self.seed_weight)

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
            
            print('inter var in replacement:')
            print('replace module:', node.name)
            print('replace in_masks:')
            print([(torch.manual_seed(100), torch.sum(torch.rand_like(v.to(torch.float)) * v))[1] for v in self.args_masks[node]])
            print('replace out_masks:')
            print([(torch.manual_seed(100), torch.sum(torch.rand_like(self.out_masks[node].to(torch.float)) * self.out_masks[node]))[1]])
            print('replace weight_mask:')
            print({k: (torch.manual_seed(100), torch.sum(torch.rand_like(v.to(torch.float)) * v))[1] for k, v in self.weight_masks[node].items()})
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

        self.inter_var_mask: Dict[id, torch.Tensor] = {}
        self.args_masks: Dict[Node, Any] = {}
        self.kwargs_masks: Dict[Node, Any] = {}
        self.out_masks: Dict[Node, Any] = {}
        self.weight_masks: Dict[Node, Dict[str, Union[torch.Tensor, torch.nn.Parameter]]] = {}
        
        self.seed_inter_var = None
        self.seed_weight = None

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

        self.logger.info('infer module masks...')
        torch.manual_seed(100)
        self.seed_inter_var = 1000
        self.seed_weight = 1000
        self.propagate_orig()
        print('inter var orig_output1:')
        print([(k, torch.sum(v) if isinstance(v, torch.Tensor) else v) for k, v in self.inter_vars.items()])
        print('inter var orig_output.grad1:')
        print([(k, v.grad is None if isinstance(v, torch.Tensor) else v) for k, v in self.inter_vars.items()])
        torch.manual_seed(100)
        self.seed_inter_var = 1000
        self.seed_weight = 1000
        self.update_direct_sparsity()
        print('inter var2:')
        print([(k, [(torch.manual_seed(100), torch.sum(torch.rand_like(i.to(torch.float)) * i))[1] if isinstance(i, torch.Tensor) else i for i in v]) for k, v in self.args_masks.items()])
        print('inter var3:')
        print([(k, (torch.manual_seed(100), torch.sum(torch.rand_like(v.to(torch.float)) * v))[1] if isinstance(v, torch.Tensor) else v) for k, v in self.out_masks.items()])
        print('inter var4:')
        print([(ko, [(ki, (torch.manual_seed(100), torch.sum(torch.rand_like(vi.to(torch.float)) * vi))[1]) for ki, vi in vo.items()]) for ko, vo in self.weight_masks.items() if ko.op == 'call_module'])
        print('inter var4.5:')
        print([(k, v.grad is None if isinstance(v, torch.Tensor) else v) for k, v in self.inter_vars.items()])
        print('inter var orig_output2:')
        print([(k, torch.sum(v) if isinstance(v, torch.Tensor) else v) for k, v in self.inter_vars.items()])
        torch.manual_seed(100)
        self.seed_inter_var = 1000
        self.seed_weight = 1000
        self.update_indirect_sparsity()
        print('inter var5:')
        print([(k, (torch.manual_seed(100), torch.sum(torch.rand_like(v.to(torch.float)) * v))[1] if isinstance(v, torch.Tensor) else v) for k, v in self.out_masks.items()])
        self.logger.info('resolve the mask conflict')

        # load the original stat dict before replace the model
        self.module.load_state_dict(self.ori_state_dict)
        self.logger.info("replace compressed modules...")
        # the mask conflict should be already resolved
        self.replace_compressed_modules()
        self.module.train(training)
        self.logger.info("speedup done")
