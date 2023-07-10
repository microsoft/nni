# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple
if TYPE_CHECKING:
    from .container import NodeInfo
    from .model_speedup import ModelSpeedup

import operator
import torch
from torch.nn import functional as F
from torch.fx.node import Node
from torch.utils._pytree import tree_flatten, tree_unflatten

from .utils import randomize_tensor_inplace, randomize_if_tensor, tree_map_zip, torch_float_dtype, poss_deepcopy


class MaskUpdater:

    def detect(self, model_speedup: 'ModelSpeedup', node: Node) -> bool:
        raise RuntimeError('detect method should be overrided!')

    def direct_update_preprocess(self, model_speedup: 'ModelSpeedup', node: Node):
        """
        preprocesses before direct update sparsity
        default action:
            do randomize to node_info.output_origin and store to node_info.output_randomize
            for submodules, randomize and apply masks to module.named_parameters
        """
        raise RuntimeError('direct_update_preprocess method should be overrided!')

    def direct_update_process(self, model_speedup: 'ModelSpeedup', node: Node):
        """
        main processes to direct update sparsity
        default action:
            get all input from node_info.output_randomize and apply the node_info.output_masks;
            execute the node and get the output;
            calc the out_mask from the output and store to node_info.output_masks.
        """
        raise RuntimeError('direct_update_process method should be overrided!')

    def direct_update_postprocess(self, model_speedup: 'ModelSpeedup', node: Node):
        """
        post processes after direct update sparsity
        default action:
            no action
        """
        raise RuntimeError('direct_update_postprocess method should be overrided!')

    def indirect_update_preprocess(self, model_speedup: 'ModelSpeedup', node: Node):
        """
        preprocesses before indirect update sparsity
        default action:
            remove all units but maintain struct of node_info.output_origin and store to node_info.output_grad
            for submodules, do tensor_requires_grad to module.named_parameters
        """
        raise RuntimeError('indirect_update_preprocess method should be overrided!')

    def indirect_update_process(self, model_speedup: 'ModelSpeedup', node: Node):
        """
        main processes to direct update sparsity
        default action:
            calc the out_mask from the node_info.output_grad and store to node_info.output_masks.
            get all input from node_info.output_origin, randomize it, apply the node_info.output_masks, and do tensor_requires_grad;
            execute the node and get the output;
            do backward to output, and for each input, store the grad to node_info.output_grad;
            for each named_parameters in submodules, update param_masks_1 from grad.
        """
        raise RuntimeError('indirect_update_process method should be overrided!')

    def indirect_update_postprocess(self, model_speedup: 'ModelSpeedup', node: Node):
        """
        post processes after indirect update sparsity
        default action:
            no action
        """
        raise RuntimeError('indirect_update_postprocess method should be overrided!')


class DefaultMaskUpdater(MaskUpdater):

    def detect(self, model_speedup: 'ModelSpeedup', node: Node) -> bool:
        """
        Return true to every node.
        """
        return True

    def direct_update_preprocess(self, model_speedup: 'ModelSpeedup', node: Node):
        """
        Do randomize to node_info.output_origin and store to node_info.output_randomize
        """
        node_info = model_speedup.node_infos[node]
        batch_dim, batch_size = model_speedup.batch_dim, model_speedup.batch_size
        node_info.output_randomize = tree_map_zip(lambda t: randomize_if_tensor(t, batch_dim, batch_size), node_info.output_origin)

    def direct_update_process(self, model_speedup: 'ModelSpeedup', node: Node):
        """
        Get all input from node_info.output_randomize and execute the node,
        calc the output_masks and store to node_info.output_masks
        """
        node_info = model_speedup.node_infos[node]
        with torch.no_grad():
            args = tree_map_zip(lambda nd: model_speedup.node_infos[nd].output_randomize if isinstance(nd, Node) else nd, node.args)
            args_masks = tree_map_zip(lambda nd: model_speedup.node_infos[nd].output_masks if isinstance(nd, Node) else None, node.args)
            args = tree_map_zip(lambda t, m: (t * m).type_as(t) if m is not None else t, args, args_masks)
            kwargs = tree_map_zip(lambda nd: model_speedup.node_infos[nd].output_randomize if isinstance(nd, Node) else nd, node.kwargs)
            kwargs_masks = tree_map_zip(lambda nd: model_speedup.node_infos[nd].output_masks if isinstance(nd, Node) else None, node.kwargs)
            kwargs = tree_map_zip(lambda t, m: (t * m).type_as(t) if m is not None else t, kwargs, kwargs_masks)

            output = getattr(model_speedup, node.op)(node.target, args, kwargs)
            if node_info.output_masks is not None:
                calc_masks = tree_map_zip(model_speedup.direct_calc_mask, output, node_info.output_masks)
            else:
                calc_masks = tree_map_zip(model_speedup.direct_calc_mask, output)

            node_info.output_masks = calc_masks

        if model_speedup.garbage_collect_values:
            # do memory collect to reduce memory usage
            for to_delete in model_speedup.user_to_last_uses.get(node, []):
                del model_speedup.node_infos[to_delete].output_randomize

    def direct_update_postprocess(self, model_speedup: 'ModelSpeedup', node: Node):
        pass

    def indirect_update_preprocess(self, model_speedup: 'ModelSpeedup', node: Node):
        node_info = model_speedup.node_infos[node]
        node_info.output_grad = tree_map_zip(lambda x: None, node_info.output_origin)

    def indirect_update_process(self, model_speedup: 'ModelSpeedup', node: Node):
        node_info = model_speedup.node_infos[node]
        batch_dim, batch_size = model_speedup.batch_dim, model_speedup.batch_size
        node_info.output_masks = tree_map_zip(model_speedup.indirect_calc_mask, node_info.output_grad, node_info.output_masks)

        def randomize_inputs(node_args):
            # output_origin -> randomize -> mask -> require_grad
            args = tree_map_zip(lambda nd: model_speedup.node_infos[nd].output_origin if isinstance(nd, Node) else nd, node_args)
            args = tree_map_zip(lambda t: randomize_if_tensor(t, batch_dim, batch_size), args)
            args_masks = tree_map_zip(lambda nd: model_speedup.node_infos[nd].output_masks if isinstance(nd, Node) else None, node_args)
            args = tree_map_zip(lambda t, m: (t * m).type_as(t) if m is not None else t, args, args_masks)

            def require_grad_(obj):
                if isinstance(obj, torch.Tensor) and model_speedup.tensor_propagate_check(obj) and obj.dtype in torch_float_dtype:
                    obj.requires_grad_(True)
                return obj

            return tree_map_zip(lambda t: require_grad_(t), args)

        # init apply input
        args = randomize_inputs(node.args)
        kwargs = randomize_inputs(node.kwargs)

        # Some operator may have the in_place operations, so we need to clone the input
        # before passing to the model_speedup.module
        args_cloned = tree_map_zip(lambda t: t.clone() if isinstance(t, torch.Tensor) else poss_deepcopy(t), args)
        kwargs_cloned = tree_map_zip(lambda t: t.clone() if isinstance(t, torch.Tensor) else poss_deepcopy(t), kwargs)

        output = getattr(model_speedup, node.op)(node.target, args_cloned, kwargs_cloned)

        tree_map_zip(model_speedup.indirect_backward, output, node_info.output_masks)

        def indirect_pass_grad(nodes, args):
            if nodes is None:
                return
            elif isinstance(nodes, (list, tuple)):
                assert isinstance(args, (list, tuple))
                for x, y in zip(nodes, args):
                    indirect_pass_grad(x, y)
            elif isinstance(nodes, dict):
                assert isinstance(args, dict)
                for x, y in zip(nodes.values(), args.values()):
                    indirect_pass_grad(x, y)
            elif isinstance(nodes, Node):
                model_speedup.indirect_pass_grad(nodes, args)
            else:
                assert not isinstance(args, torch.Tensor)

        indirect_pass_grad(node.args, args)
        indirect_pass_grad(node.kwargs, kwargs)

    def indirect_update_postprocess(self, model_speedup: 'ModelSpeedup', node: Node):
        pass


class LeafModuleMaskUpdater(DefaultMaskUpdater):

    def detect(self, model_speedup: 'ModelSpeedup', node: Node) -> bool:
        """
        the default MaskUpdater for leaf module, so return true if the node is a module calling
        """
        if node.op == 'call_module':
            module: torch.nn.Module = model_speedup.fetch_attr(node.target)
            param_masks = model_speedup.masks.get(node.target, {})
            for k, v in module.named_parameters():
                if k not in param_masks:
                    param_masks[k] = torch.ones_like(v)
            model_speedup.node_infos[node].module = module
            model_speedup.node_infos[node].param_masks = param_masks
            return True
        else:
            return False

    def direct_update_preprocess(self, model_speedup: 'ModelSpeedup', node: Node):
        super().direct_update_preprocess(model_speedup, node)

        with torch.no_grad():
            node_info: 'NodeInfo' = model_speedup.node_infos[node]
            for k, v in node_info.module.named_parameters():
                randomize_tensor_inplace(v)
                v *= node_info.param_masks[k] # in-place addition

    def indirect_update_preprocess(self, model_speedup: 'ModelSpeedup', node: Node):
        super().indirect_update_preprocess(model_speedup, node)

        node_info: 'NodeInfo' = model_speedup.node_infos[node]
        for _, v in node_info.module.named_parameters():
            if isinstance(v, torch.Tensor) and model_speedup.tensor_propagate_check(v) and v.dtype in torch_float_dtype:
                    v.requires_grad_(True)

    def indirect_update_process(self, model_speedup: 'ModelSpeedup', node: Node):
        super().indirect_update_process(model_speedup, node)

        # update the sparsity of the paramters
        node_info: 'NodeInfo' = model_speedup.node_infos[node]
        for k, v in node_info.module.named_parameters():
            if isinstance(v, torch.Tensor) and model_speedup.tensor_propagate_check(v) and v.dtype in torch_float_dtype:
                grad_zero = v.grad.data == 0
                node_info.param_masks[k][grad_zero] = 0


class NoMaskUpdater(DefaultMaskUpdater):
    """
    For some ops that will not produce masks.
    """
    def detect(self, model_speedup: 'ModelSpeedup', node: Node) -> bool:
        """
        the default MaskUpdater for operators that will not change mask value
        """
        if node.op == 'call_function':
            if node.target in (len, operator.is_, operator.is_not, operator.contains):
                return True
        elif node.op == 'call_method':
            if isinstance(node.args[0], Node) and isinstance(model_speedup.node_infos[node.args[0]].output_origin, torch.Tensor):
                if node.target in ('dim', 'size'):
                    return True
        return False

    def direct_update_process(self, model_speedup: 'ModelSpeedup', node: Node):
        """
        get all input from node_info.output_randomize and execute the node
        calc the out_mask and store to node_info.output_masks
        """
        with torch.no_grad():
            model_speedup.node_infos[node].output_masks = tree_map_zip(lambda t: None, model_speedup.node_infos[node].output_origin)

        if model_speedup.garbage_collect_values:
            # do memory collect to reduce memory usage
            for to_delete in model_speedup.user_to_last_uses.get(node, []):
                del model_speedup.node_infos[to_delete]._output_randomize


# in all the following function, the first arg name is `input`, and don't have other tensor as input args.
no_change_act_func = (
    F.relu,
    F.relu_,
    F.hardtanh,
    F.hardtanh_,
    F.hardswish,
    F.relu6,
    F.elu,
    F.elu_,
    F.selu,
    F.celu,
    F.leaky_relu,
    F.leaky_relu_,
    # F.prelu,  \\this need more support
    F.rrelu,
    F.rrelu_,
    # F.glu,  \\this need more support
    F.gelu,
    F.logsigmoid,
    F.hardshrink,
    F.tanhshrink,
    F.softsign,
    F.softplus,
    F.softmin,
    F.softmax,
    F.softshrink,
    F.gumbel_softmax,
    F.log_softmax,
    F.tanh,
    F.sigmoid,
    F.hardsigmoid,
    F.silu,
    F.mish,
    # F.batch_norm,  \\this need more support
    # F.group_norm,  \\this need more support
    # F.instance_norm,  \\this need more support
    # F.layer_norm,  \\this need more support
    # F.local_response_norm,  \\this need more support
    # F.normalize  \\this need more support
)

no_change_act_module = (
    torch.nn.Softmin,
    torch.nn.Softmax,
    torch.nn.Softmax2d,
    torch.nn.LogSoftmax,
    # torch.nn.AdaptiveLogSoftmaxWithLoss,  \\need test
    torch.nn.ELU,
    torch.nn.Hardshrink,
    torch.nn.Hardsigmoid,
    torch.nn.Hardtanh,
    torch.nn.Hardswish,
    torch.nn.LeakyReLU,
    torch.nn.LogSigmoid,
    # torch.nn.MultiheadAttention,  \\this need more support
    # torch.nn.PReLU,
    torch.nn.ReLU,
    torch.nn.ReLU6,
    torch.nn.RReLU,
    torch.nn.SELU,
    torch.nn.CELU,
    torch.nn.GELU,
    torch.nn.Sigmoid,
    torch.nn.SiLU,
    torch.nn.Mish,
    torch.nn.Softplus,
    torch.nn.Softshrink,
    torch.nn.Softsign,
    torch.nn.Tanh,
    torch.nn.Tanhshrink,
    # torch.nn.GLU  \\this need more support
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.LazyBatchNorm1d,
    torch.nn.LazyBatchNorm2d,
    torch.nn.LazyBatchNorm3d,
    torch.nn.GroupNorm,
    torch.nn.SyncBatchNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
    torch.nn.LazyInstanceNorm1d,
    torch.nn.LazyInstanceNorm2d,
    torch.nn.LazyInstanceNorm3d,
    torch.nn.LayerNorm,
)


class NoChangeMaskUpdater(DefaultMaskUpdater):
    """
    for some special op that masks will not change when execute
    1. for getitem op, it's no need to calc masks. do in fast path to run the algorithm faster.
    2. for (softmax, log_softmax) ops, the default process will get a wrong mask. actually we should just copy the mask from input to
        output.
    """
    def __init__(self, customized_no_change_act_module: Tuple | None = None,
                 customized_no_change_act_func: Tuple | None = None):
        self.no_change_act_module = no_change_act_module if not customized_no_change_act_module \
            else (no_change_act_module + customized_no_change_act_module)
        self.no_change_act_func = no_change_act_func if not customized_no_change_act_func \
            else (no_change_act_func + customized_no_change_act_func)

    def direct_activation(self, model_speedup: 'ModelSpeedup', node: Node):
        if len(node.args) != 0:
            input_node = node.args[0]
        else:
            input_node = node.kwargs['input']
        input_mask = model_speedup.node_infos[input_node].output_masks
        model_speedup.node_infos[node].output_masks = \
            tree_map_zip(lambda t: t.clone().detach() if isinstance(t, torch.Tensor) else poss_deepcopy(t), input_mask)

    def indirect_activation(self, model_speedup: 'ModelSpeedup', node: Node):
        if len(node.args) != 0:
            input_node = node.args[0]
        else:
            input_node = node.kwargs['input']

        input_grad = tree_map_zip(lambda t, m: (t * m).type_as(t) if isinstance(m, torch.Tensor) else t, \
            model_speedup.node_infos[node].output_grad, model_speedup.node_infos[node].output_masks)
        dummy_input = torch.rand_like(input_grad)
        dummy_input.grad = input_grad
        model_speedup.indirect_pass_grad(input_node, dummy_input)

    def direct_getitem(self, model_speedup: 'ModelSpeedup', node: Node):
        assert len(node.args) == 2
        arg_0_masks = model_speedup.node_infos[node.args[0]].output_masks
        arg_1_val = model_speedup.node_infos[node.args[1]].output_randomize if isinstance(node.args[1], Node) else node.args[1]
        sub_mask = operator.getitem(arg_0_masks, arg_1_val)

        model_speedup.node_infos[node].output_masks = \
            tree_map_zip(lambda t: t.clone().detach() if isinstance(t, torch.Tensor) else poss_deepcopy(t), sub_mask)

    def indirect_getitem(self, model_speedup: 'ModelSpeedup', node: Node):
        assert len(node.args) == 2
        input_grad = tree_map_zip(lambda t, m: (t * m).type_as(t) if isinstance(m, torch.Tensor) else t, \
            model_speedup.node_infos[node].output_grad, model_speedup.node_infos[node].output_masks)
        arg_1_val = model_speedup.node_infos[node.args[1]].output_randomize if isinstance(node.args[1], Node) else node.args[1]

        input_node_info = model_speedup.node_infos[node.args[0]]
        flat_args, spec = tree_flatten(input_node_info.output_grad)
        flat_grads = [None for _ in range(len(flat_args))]
        flat_grads[arg_1_val] = input_grad
        input_grads = tree_unflatten(flat_grads, spec)

        def add_grad(grad, input_grad):
            if isinstance(input_grad, torch.Tensor):
                if grad is not None and input_grad is not None:
                    return grad + input_grad
                elif grad is None:
                    return input_grad
                else:
                    return grad
            else:
                return grad

        model_speedup.node_infos[node].output_grad = tree_map_zip(add_grad, model_speedup.node_infos[node.args[0]].output_grad, input_grads)

    def detect(self, model_speedup: 'ModelSpeedup', node: Node) -> bool:
        return self.detect_helper(model_speedup, node) is not None

    def detect_helper(self, model_speedup: 'ModelSpeedup', node: Node):
        if node.op == 'call_function':
            if node.target in self.no_change_act_func:
                return self.direct_activation, self.indirect_activation
            elif node.target == operator.getitem:
                if isinstance(node.args[0], Node) and type(model_speedup.node_infos[node.args[0]].output_origin) in (tuple, list, dict):
                    return self.direct_getitem, self.indirect_getitem
        elif node.op == 'call_module':
            module: torch.nn.Module = model_speedup.fetch_attr(node.target)
            if isinstance(module, self.no_change_act_module):
                return self.direct_activation, self.indirect_activation
        elif node.op == 'call_method':
            if isinstance(node.args[0], Node) and isinstance(model_speedup.node_infos[node.args[0]].output_origin, torch.Tensor):
                if node.target in ('clone', 'detach'):
                    return self.direct_activation, self.indirect_activation
        return None

    def direct_update_process(self, model_speedup: 'ModelSpeedup', node: Node):
        """
        get all input from node_info.output_randomize and execute the node
        calc the out_mask and store to node_info.output_masks
        """
        direct_fn, _ = self.detect_helper(model_speedup, node)
        with torch.no_grad():
            direct_fn(model_speedup, node)

        if model_speedup.garbage_collect_values:
            # do memory collect to reduce memory usage
            for to_delete in model_speedup.user_to_last_uses.get(node, []):
                del model_speedup.node_infos[to_delete].output_randomize

    def indirect_update_process(self, model_speedup: 'ModelSpeedup', node: Node):
        node_info = model_speedup.node_infos[node]
        node_info.output_masks = tree_map_zip(model_speedup.indirect_calc_mask, node_info.output_grad, node_info.output_masks)

        _, indirect_fn = self.detect_helper(model_speedup, node)
        indirect_fn(model_speedup, node)
