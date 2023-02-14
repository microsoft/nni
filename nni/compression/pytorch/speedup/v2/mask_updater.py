# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .model_speedup import ModelSpeedup

import operator
import torch
from torch import nn
from torch.nn import functional as F
from torch.fx.node import Node

from .container import NodeInfo
from .utils import randomize_tensor_inplace, randomize_if_tensor, tree_map_zip, torch_float_dtype


class MaskUpdater:

    def detect(self, model_speedup: 'ModelSpeedup', node: Node) -> bool:
        raise RuntimeError('detect method should be overrided!')

    def direct_update_preprocess(self, model_speedup: 'ModelSpeedup', node: Node):
        """
        preprocesses before direct update sparsity
        default action:
            do randomize to slot.value_0 and store to slot.value_2
            for submodules, randomize and apply masks to module.named_parameters
        """
        raise RuntimeError('direct_update_preprocess method should be overrided!')

    def direct_update_process(self, model_speedup: 'ModelSpeedup', node: Node):
        """
        main processes to direct update sparsity
        default action:
            get all input from slot.value_2 and apply the slot.mask_1;
            execute the node and get the output;
            calc the out_mask from the output and store to slot.mask_1.
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
            remove all units but maintain struct of slot.value_0 and store to slot.value_3
            for submodules, do tensor_requires_grad to module.named_parameters
        """
        raise RuntimeError('indirect_update_preprocess method should be overrided!')

    def indirect_update_process(self, model_speedup: 'ModelSpeedup', node: Node):
        """
        main processes to direct update sparsity
        default action:
            calc the out_mask from the slot.value_3 and store to slot.mask_2.
            get all input from slot.value_0, randomize it, apply the slot.mask_1, and do tensor_requires_grad;
            execute the node and get the output;
            do backward to output, and for each input, store the grad to slot.value_3;
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
            args = tree_map_zip(lambda t, m: t * m if m is not None else t, args, args_masks)
            kwargs = tree_map_zip(lambda nd: model_speedup.node_infos[nd].output_randomize if isinstance(nd, Node) else nd, node.kwargs)
            kwargs_masks = tree_map_zip(lambda nd: model_speedup.node_infos[nd].output_masks if isinstance(nd, Node) else None, node.kwargs)
            kwargs = tree_map_zip(lambda t, m: t * m if m is not None else t, kwargs, kwargs_masks)

            output = getattr(model_speedup, node.op)(node.target, args, kwargs)
            if node_info.output_masks is not None:
                calc_masks = tree_map_zip(model_speedup.direct_calc_mask, output, node_info.output_masks)
            else:
                calc_masks = tree_map_zip(model_speedup.direct_calc_mask, output)

            node_info.output_masks = calc_masks

        if model_speedup.garbage_collect_values:
            # do memory collect to reduce memory usage
            for to_delete in model_speedup.user_to_last_uses.get(node, []):
                del model_speedup.node_infos[to_delete]._output_randomize

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
            args = tree_map_zip(lambda t, m: t * m if m is not None else t, args, args_masks)

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
        args_cloned = tree_map_zip(lambda t: t.clone() if isinstance(t, torch.Tensor) else t, args)
        kwargs_cloned = tree_map_zip(lambda t: t.clone() if isinstance(t, torch.Tensor) else t, kwargs)

        output = getattr(model_speedup, node.op)(node.target, args_cloned, kwargs_cloned)

        tree_map_zip(model_speedup.indirect_backward, output, node_info.output_masks)
        args_node_infos = tree_map_zip(lambda nd: model_speedup.node_infos[nd] if isinstance(nd, Node) else None, node.args)
        kwargs_node_infos = tree_map_zip(lambda nd: model_speedup.node_infos[nd] if isinstance(nd, Node) else None, node.kwargs)

        def indirect_pass_grad(node_args, args):
            if node_args is None:
                return
            elif isinstance(node_args, (list, tuple)):
                assert isinstance(args, (list, tuple))
                for x, y in zip(node_args, args):
                    indirect_pass_grad(x, y)
            elif isinstance(node_args, dict):
                assert isinstance(args, dict)
                for x, y in zip(node_args.values(), args.values()):
                    indirect_pass_grad(x, y)
            elif isinstance(node_args, NodeInfo):
                model_speedup.indirect_pass_grad(node_args, args)
            else:
                raise RuntimeError(f'Type {type(node_args)} is not supported during indirect pass grad.')

        indirect_pass_grad(args_node_infos, args)
        indirect_pass_grad(kwargs_node_infos, kwargs)

    def indirect_update_postprocess(self, model_speedup: 'ModelSpeedup', node: Node):
        pass


class LeafModuleMaskUpdater(DefaultMaskUpdater):

    def detect(self, model_speedup: 'ModelSpeedup', node: Node) -> bool:
        """
        the default MaskUpdater for leaf module, so return true if the node is a module calling
        """
        if node.op == 'call_module':
            module: torch.nn.Module = model_speedup.fetch_attr(node.target)
            param_masks = model_speedup.masks_file.get(node.target, {})
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
                if node.target in ('dim', 'size', 'clone', 'detach'):
                    return True
        return False

    def direct_update_process(self, model_speedup: 'ModelSpeedup', node: Node):
        """
        get all input from slot.value_2 and execute the node
        calc the out_mask and store to slot.mask_1
        """
        with torch.no_grad():
            model_speedup.node_infos[node].output_masks = tree_map_zip(lambda t: None, model_speedup.node_infos[node].output_origin)

        if model_speedup.garbage_collect_values:
            # do memory collect to reduce memory usage
            for to_delete in model_speedup.user_to_last_uses.get(node, []):
                del model_speedup.node_infos[to_delete]._output_randomize


class NoChangeMaskUpdater(DefaultMaskUpdater):
    """
    for some special op that masks will not change when execute
    1. for getitem op, it's no need to calc masks. do in fast path to run the algorithm faster.
    2. for (softmax, log_softmax) ops, the default process will get a wrong mask. actually we should just copy the mask from input to
        output.
    """
    def direct_softmax(self, model_speedup: 'ModelSpeedup', node: Node):
        if len(node.args) != 0:
            input_node = node.args[0]
        else:
            input_node = node.kwargs['input']
        input_mask = model_speedup.node_infos[input_node].output_masks
        model_speedup.node_infos[node].output_masks = \
            tree_map_zip(lambda t: t.clone().detach() if isinstance(t, torch.Tensor) else t, input_mask)

    def indirect_softmax(self, model_speedup: 'ModelSpeedup', node: Node):
        if len(node.args) != 0:
            input_node = node.args[0]
        else:
            input_node = node.kwargs['input']

        input_grad = tree_map_zip(lambda t, m: t * m if isinstance(m, torch.Tensor) else t, \
            model_speedup.node_infos[node].output_grad, model_speedup.node_infos[node].output_masks)
        model_speedup.indirect_pass_grad(input_node, input_grad)

    def direct_getitem(self, model_speedup: 'ModelSpeedup', node: Node):
        assert len(node.args) == 2
        arg_0_masks = model_speedup.node_infos[node.args[0]].output_masks
        arg_1_val = model_speedup.node_infos[node.args[1]].output_randomize
        sub_mask = operator.getitem(arg_0_masks, arg_1_val)

        model_speedup.node_infos[node].output_masks = \
            tree_map_zip(lambda t: t.clone().detach() if isinstance(t, torch.Tensor) else t, sub_mask)

    def indirect_getitem(self, model_speedup: 'ModelSpeedup', node: Node):
        assert len(node.args) == 2
        input_grad = tree_map_zip(lambda t, m: t * m if isinstance(m, torch.Tensor) else t, \
            model_speedup.node_infos[node].output_grad, model_speedup.node_infos[node].output_masks)
        model_speedup.indirect_pass_grad(node.args[0], input_grad)

    def detect(self, model_speedup: 'ModelSpeedup', node: Node) -> bool:
        return self.detect_helper(model_speedup, node) is not None

    def detect_helper(self, model_speedup: 'ModelSpeedup', node: Node):
        if node.op == 'call_function':
            if node.target in (F.log_softmax, F.softmax):
                return self.direct_softmax, self.indirect_softmax
            elif node.target == operator.getitem:
                if isinstance(node.args[0], Node) and type(model_speedup.node_infos[node.args[0]].output_origin) in (tuple, list, dict):
                    return self.direct_getitem, self.indirect_getitem
        elif node.op == 'call_module':
            module: torch.nn.Module = model_speedup.fetch_attr(node.target)
            if isinstance(module, (nn.LogSoftmax, nn.Softmax)):
                return self.direct_softmax, self.indirect_softmax
        return None

    def direct_update_process(self, model_speedup: 'ModelSpeedup', node: Node):
        """
        get all input from slot.value_2 and execute the node
        calc the out_mask and store to slot.mask_1
        """
        direct_fn, _ = self.detect_helper(model_speedup, node)
        with torch.no_grad():
            direct_fn(model_speedup, node)

        if model_speedup.garbage_collect_values:
            # do memory collect to reduce memory usage
            for to_delete in model_speedup.user_to_last_uses.get(node, []):
                del model_speedup.node_infos[to_delete]._output_randomize

    def indirect_update_process(self, model_speedup: 'ModelSpeedup', node: Node):
        node_info = model_speedup.node_infos[node]
        node_info.output_masks = tree_map_zip(model_speedup.indirect_calc_mask, node_info.output_grad, node_info.output_masks)

        _, indirect_fn = self.detect_helper(model_speedup, node)
        indirect_fn(model_speedup, node)
