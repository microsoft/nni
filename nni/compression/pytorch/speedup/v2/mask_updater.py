import operator
import torch
from torch import nn
from torch.nn import functional as F
from torch.fx.node import Node
from nni.common.concrete_trace_utils.utils import map_recursive, map_recursive_zip
from nni.compression.pytorch.utils.utils import randomize_tensor
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .compressor import ModelSpeedup
    from .container import NodeInfo

class MaskUpdater:
    def detect(self, model_speedup: 'ModelSpeedup', node: Node) -> bool:
        raise RuntimeError('detect method should be overrided!')

    @staticmethod
    def propagate_originally(model_speedup: 'ModelSpeedup', node: Node):
        """
        propagate normally to get informations of intermediate variables such as shape, dtype of tensors
        default action:
            execute and store values to slot.value_0(intermediate variables when assigned) and slot.value_1(intermediate variables after in-place ops)
        """
        args, kwargs = node.args, node.kwargs
        args = map_recursive(model_speedup.slot_getter_value_1, args)
        args = map_recursive(model_speedup.tensor_detacher, args)
        kwargs = map_recursive(model_speedup.slot_getter_value_1, kwargs)
        kwargs = map_recursive(model_speedup.tensor_detacher, kwargs)

        output = getattr(model_speedup, node.op)(node.target, args, kwargs)

        model_speedup.slots[node].value_0 = output
        model_speedup.slots[node].status['value_0'] += 1
        model_speedup.slots[node].value_1 = map_recursive(model_speedup.tensor_clone_detacher, output)
        model_speedup.slots[node].status['value_1'] += 1

        if model_speedup.garbage_collect_values:
            # do memory collect to reduce memory usage
            for to_delete in model_speedup.user_to_last_uses.get(node, []):
                model_speedup.slots[to_delete].value_1 = None
            pass

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
        the default MaskUpdater, so return true to every node
        """
        return True

    def direct_update_preprocess(self, model_speedup: 'ModelSpeedup', node: Node):
        """
        do randomize to slot.value_0 and store to slot.value_2
        """
        model_speedup.slots[node].value_2 = map_recursive(model_speedup.tensor_randomizer, model_speedup.slots[node].value_0)
        model_speedup.slots[node].status['value_2'] += 1

    def direct_update_process(self, model_speedup: 'ModelSpeedup', node: Node):
        """
        get all input from slot.value_2 and execute the node
        calc the out_mask and store to slot.mask_1
        """
        with torch.no_grad():
            args_2 = map_recursive(model_speedup.slot_getter_value_2, node.args)
            arg_masks = map_recursive(model_speedup.slot_getter_mask_1, node.args)
            args_2 = map_recursive_zip(model_speedup.mask_applier, args_2, arg_masks)
            kwargs_2 = map_recursive(model_speedup.slot_getter_value_2, node.kwargs)
            kwarg_masks = map_recursive(model_speedup.slot_getter_mask_1, node.kwargs)
            kwargs_2 = map_recursive_zip(model_speedup.mask_applier, kwargs_2, kwarg_masks)

            output = getattr(model_speedup, node.op)(node.target, args_2, kwargs_2)

            model_speedup.slots[node].mask_1 = map_recursive(model_speedup.direct_calc_mask, output)
            model_speedup.slots[node].status['mask_1'] += 1

        if model_speedup.garbage_collect_values:
            # do memory collect to reduce memory usage
            # model_speedup.slots[node].value_2 = None
            pass

    def direct_update_postprocess(self, model_speedup: 'ModelSpeedup', node: Node):
        pass

    def indirect_update_preprocess(self, model_speedup: 'ModelSpeedup', node: Node):
        model_speedup.slots[node].value_3 = map_recursive(model_speedup.tensor_deleter, model_speedup.slots[node].value_0)
        model_speedup.slots[node].status['value_3'] += 1

    def indirect_update_process(self, model_speedup: 'ModelSpeedup', node: Node):
        output_3 = map_recursive(model_speedup.slot_getter_value_3, node)
        output_masks_1 = map_recursive(model_speedup.slot_getter_mask_1, node)
        output_masks_2 = map_recursive_zip(model_speedup.indirect_calc_mask2, output_masks_1, output_3)

        model_speedup.slots[node].mask_2 = output_masks_2
        model_speedup.slots[node].status['mask_2'] += 1

        # init apply input
        # randomized, so it's same to use slot_getter_value_orig or slot_getter_value_orig_inplace
        args = map_recursive(model_speedup.slot_getter_value_0, node.args)
        args_rand = map_recursive(model_speedup.tensor_randomizer, args)
        arg_masks = map_recursive(model_speedup.slot_getter_mask_1, node.args)
        args_rand = map_recursive_zip(model_speedup.mask_applier, args_rand, arg_masks)
        map_recursive(model_speedup.tensor_requires_grad, args_rand)

        kwargs = map_recursive(model_speedup.slot_getter_value_0, node.kwargs)
        kwargs_rand = map_recursive(model_speedup.tensor_randomizer, kwargs)
        kwarg_masks = map_recursive(model_speedup.slot_getter_mask_1, node.kwargs)
        kwargs_rand = map_recursive_zip(model_speedup.mask_applier, kwargs_rand, kwarg_masks)
        map_recursive(model_speedup.tensor_requires_grad, kwargs_rand)

        # Some operator may have the in_place operations, so we need to clone the input
        # before passing to the model_speedup.module
        args_rand_cloned = map_recursive(model_speedup.tensor_cloner, args_rand)
        kwargs_rand_cloned = map_recursive(model_speedup.tensor_cloner, kwargs_rand)

        output = getattr(model_speedup, node.op)(node.target, args_rand_cloned, kwargs_rand_cloned)

        map_recursive_zip(model_speedup.indirect_update_backward, output, output_masks_2)

        arg_values_3 = map_recursive(model_speedup.slot_getter_value_3, node.args)
        kwarg_values_3 = map_recursive(model_speedup.slot_getter_value_3, node.kwargs)

        map_recursive_zip(model_speedup.indirect_pass_grad, arg_values_3, args_rand)
        map_recursive_zip(model_speedup.indirect_pass_grad, kwarg_values_3, kwargs_rand)

        if model_speedup.garbage_collect_values:
            # do memory collect to reduce memory usage
            model_speedup.slots[node].mask_1 = None
            pass

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
            model_speedup.node_infos[node].param_masks_0 = param_masks
            return True
        else:
            return False

    def direct_update_preprocess(self, model_speedup: 'ModelSpeedup', node: Node):
        super().direct_update_preprocess(model_speedup, node)

        with torch.no_grad():
            node_info: 'NodeInfo' = model_speedup.node_infos[node]

            for k, v in node_info.module.named_parameters():
                randomize_tensor(v, model_speedup.randomize_range_float[0], model_speedup.randomize_range_float[1])
                v *= node_info.param_masks_0[k] # in-place addition

    def indirect_update_preprocess(self, model_speedup: 'ModelSpeedup', node: Node):
        super().indirect_update_preprocess(model_speedup, node)

        node_info: 'NodeInfo' = model_speedup.node_infos[node]

        for _k, v in node_info.module.named_parameters():
            model_speedup.tensor_requires_grad(v)

    def indirect_update_process(self, model_speedup: 'ModelSpeedup', node: Node):
        super().indirect_update_process(model_speedup, node)

        # update the sparsity of the paramters
        node_info: 'NodeInfo' = model_speedup.node_infos[node]
        for k, v in node_info.module.named_parameters():
            grad_zero = v.grad.data == 0
            node_info.param_masks_1[k] = node_info.param_masks_0[k].clone()
            node_info.param_masks_1[k][grad_zero] = 0

class NoMaskUpdater(DefaultMaskUpdater):
    """
    for some ops that will not produce masks.
    """
    def detect(self, model_speedup: 'ModelSpeedup', node: Node) -> bool:
        """
        the default MaskUpdater for operators that will not change mask value
        """
        if node.op == 'call_function':
            if node.target in (len, operator.is_, operator.is_not, operator.contains):
                return True
        elif node.op == 'call_method':
            if isinstance(node.args[0], Node) and isinstance(model_speedup.slots[node.args[0]].value_0, torch.Tensor):
                if node.target in ('dim', 'size', 'clone', 'detach'):
                    return True
        return False

    def direct_update_process(self, model_speedup: 'ModelSpeedup', node: Node):
        """
        get all input from slot.value_2 and execute the node
        calc the out_mask and store to slot.mask_1
        """
        with torch.no_grad():
            model_speedup.slots[node].mask_1 = map_recursive(model_speedup.tensor_deleter, model_speedup.slots[node].value_0)
            model_speedup.slots[node].status['mask_1'] += 1

        if model_speedup.garbage_collect_values:
            # do memory collect to reduce memory usage
            # model_speedup.slots[node].value_2 = None
            pass

class NoChangeMaskUpdater(DefaultMaskUpdater):
    """
    """
    def direct_softmax(self, model_speedup: 'ModelSpeedup', node: Node):
        if len(node.args) != 0:
            input_node = node.args[0]
        else:
            input_node = node.kwargs['input']
        input_mask = model_speedup.slot_getter_mask_1(input_node)
        model_speedup.slots[node].mask_1 = map_recursive(model_speedup.tensor_clone_detacher, input_mask)
        model_speedup.slots[node].status['mask_1'] += 1

    def indirect_softmax(self, model_speedup: 'ModelSpeedup', node: Node):
        if len(node.args) != 0:
            input_node = node.args[0]
        else:
            input_node = node.kwargs['input']
        output_3 = map_recursive(model_speedup.slot_getter_value_3, node)
        input_values_3 = map_recursive(model_speedup.slot_getter_value_3, input_node)
        map_recursive_zip(model_speedup.indirect_pass_grad, input_values_3, output_3)

    def direct_getitem(self, model_speedup: 'ModelSpeedup', node: Node):
        assert len(node.args) == 2
        arg0_mask = map_recursive(model_speedup.slot_getter_mask_1, node.args[0])
        attrname = map_recursive(model_speedup.slot_getter_value_2, node.args[1])

        sub_mask = operator.getitem(arg0_mask, attrname)

        model_speedup.slots[node].mask_1 = map_recursive(model_speedup.tensor_clone_detacher, sub_mask)
        model_speedup.slots[node].status['mask_1'] += 1

    def indirect_getitem(self, model_speedup: 'ModelSpeedup', node: Node):
        assert len(node.args) == 2
        arg0_values_3 = map_recursive(model_speedup.slot_getter_value_3, node.args[0])
        attrname = map_recursive(model_speedup.slot_getter_value_2, node.args[1])
        sub_value_3 = operator.getitem(arg0_values_3, attrname)

        output_3 = map_recursive(model_speedup.slot_getter_value_3, node)
        map_recursive_zip(model_speedup.indirect_pass_grad, sub_value_3, output_3)

    def detect(self, model_speedup: 'ModelSpeedup', node: Node) -> bool:
        """
        the default MaskUpdater for operators that will not change mask value
        """
        if node.op == 'call_function':
            if node.target in (F.log_softmax, F.softmax):
                model_speedup.node_infos[node].infos['direct'] = self.direct_softmax
                model_speedup.node_infos[node].infos['indirect'] = self.indirect_softmax
                return True
            elif node.target == operator.getitem:
                if isinstance(node.args[0], Node) and type(model_speedup.slots[node.args[0]].value_0) in (tuple, list, dict):
                    model_speedup.node_infos[node].infos['direct'] = self.direct_getitem
                    model_speedup.node_infos[node].infos['indirect'] = self.indirect_getitem
                    return True
        elif node.op == 'call_module':
            module: torch.nn.Module = model_speedup.fetch_attr(node.target)
            if isinstance(module, (nn.LogSoftmax, nn.Softmax)):
                model_speedup.node_infos[node].infos['direct'] = self.direct_softmax
                model_speedup.node_infos[node].infos['indirect'] = self.indirect_softmax
                return True
        return False

    def direct_update_process(self, model_speedup: 'ModelSpeedup', node: Node):
        """
        get all input from slot.value_2 and execute the node
        calc the out_mask and store to slot.mask_1
        """
        with torch.no_grad():
            model_speedup.node_infos[node].infos['direct'](model_speedup, node)

        if model_speedup.garbage_collect_values:
            # do memory collect to reduce memory usage
            # model_speedup.slots[node].value_2 = None
            pass

    def indirect_update_process(self, model_speedup: 'ModelSpeedup', node: Node):
        output_3 = map_recursive(model_speedup.slot_getter_value_3, node)
        output_masks_1 = map_recursive(model_speedup.slot_getter_mask_1, node)
        output_masks_2 = map_recursive_zip(model_speedup.indirect_calc_mask2, output_masks_1, output_3)

        model_speedup.slots[node].mask_2 = output_masks_2
        model_speedup.slots[node].status['mask_2'] += 1

        model_speedup.node_infos[node].infos['indirect'](model_speedup, node)

        if model_speedup.garbage_collect_values:
            # do memory collect to reduce memory usage
            model_speedup.slots[node].mask_1 = None
            pass
