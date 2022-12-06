import torch
from torch.fx.node import Node
from nni.common.concrete_trace_utils.utils import run_onlyif_instance, map_recursive, map_recursive_zip
from nni.compression.pytorch.utils.utils import (rand_like_with_shape,
                                                 randomize_tensor,
                                                 torch_float_dtype,
                                                 torch_integer_dtype)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .compressor import ModelSpeedup

class MaskUpdater:
    def detect(self, model_speedup: 'ModelSpeedup', node: Node) -> bool:
        raise RuntimeError('detect method should be overrided!')

    def direct_update_preprocess(self, model_speedup: 'ModelSpeedup', node: Node) -> bool:
        raise RuntimeError('direct_update_preprocess method should be overrided!')

    def direct_update_process(self, model_speedup: 'ModelSpeedup', node: Node) -> bool:
        raise RuntimeError('direct_update_process method should be overrided!')

    def direct_update_postprocess(self, model_speedup: 'ModelSpeedup', node: Node) -> bool:
        raise RuntimeError('direct_update_postprocess method should be overrided!')

    def indirect_update_preprocess(self, model_speedup: 'ModelSpeedup', node: Node) -> bool:
        raise RuntimeError('indirect_update_preprocess method should be overrided!')

    def indirect_update_process(self, model_speedup: 'ModelSpeedup', node: Node) -> bool:
        raise RuntimeError('indirect_update_process method should be overrided!')

    def indirect_update_postprocess(self, model_speedup: 'ModelSpeedup', node: Node) -> bool:
        raise RuntimeError('indirect_update_postprocess method should be overrided!')

class DefaultMaskUpdater(MaskUpdater):

    def detect(self, model_speedup: 'ModelSpeedup', node: Node) -> bool:
        """
        the default MaskUpdater, so return true to every node
        """
        model_speedup.node_infos[node].mask_updater = self
        return True

    def direct_update_preprocess(self, model_speedup: 'ModelSpeedup', node: Node) -> bool:
        """
        do randomize to slot.value_0 and store to slot.value_2
        """
        model_speedup.slots[node].value_2 = map_recursive(model_speedup.tensor_randomizer, model_speedup.slots[node].value_0)
        model_speedup.slots[node].status['value_2'] += 1

    def direct_update_process(self, model_speedup: 'ModelSpeedup', node: Node) -> bool:
        """
        get all input from slot.value_2 and execute the node
        calc the out_mask and store to slot.mask_1
        """
        with torch.no_grad():
            # if node.op == 'call_module':
            #     node_info: NodeInfo = model_speedup.node_infos[node]
            #     sub_module: nn.Module = model_speedup.fetch_attr(node.target)

            #     for _k, v in sub_module.named_parameters():
            #         randomize_tensor(v, model_speedup.randomize_range_float[0], model_speedup.randomize_range_float[1])

            #     for k, v in sub_module.named_parameters():
            #         v *= node_info.param_masks_0[k] # in-place addition

            args = map_recursive(model_speedup.slot_getter_value_2, node.args)
            arg_masks = map_recursive(model_speedup.slot_getter_mask_1, node.args)
            args = map_recursive_zip(model_speedup.mask_applier, args, arg_masks)
            kwargs = map_recursive(model_speedup.slot_getter_value_2, node.kwargs)
            kwarg_masks = map_recursive(model_speedup.slot_getter_mask_1, node.kwargs)
            kwargs = map_recursive_zip(model_speedup.mask_applier, kwargs, kwarg_masks)

            output = getattr(model_speedup, node.op)(node.target, args, kwargs)

            model_speedup.slots[node].mask_1 = map_recursive(model_speedup.direct_calc_mask, output)
            model_speedup.slots[node].status['mask_1'] += 1

            # do memory collect / compression

    def direct_update_postprocess(self, model_speedup: 'ModelSpeedup', node: Node) -> bool:
        pass

    def indirect_update_preprocess(self, model_speedup: 'ModelSpeedup', node: Node) -> bool:
        pass

    def indirect_update_process(self, model_speedup: 'ModelSpeedup', node: Node) -> bool:
        output = map_recursive(model_speedup.slot_getter_value_2, node)
        output_masks_1 = map_recursive(model_speedup.slot_getter_mask_1, node)
        output_masks_2 = map_recursive_zip(model_speedup.indirect_calc_mask, output_masks_1, output)

        model_speedup.slots[node].mask_2 = output_masks_2
        model_speedup.slots[node].status['mask_2'] += 1

        # init apply input
        # randomized, so it's same to use slot_getter_value_orig or slot_getter_value_orig_inplace
        args = map_recursive(model_speedup.slot_getter_value_0, node.args)
        args = map_recursive(model_speedup.tensor_randomizer, args)
        arg_masks = map_recursive(model_speedup.slot_getter_mask_1, node.args)
        args = map_recursive_zip(model_speedup.mask_applier, args, arg_masks)
        map_recursive(model_speedup.tensor_requires_grad, args)

        kwargs = map_recursive(model_speedup.slot_getter_value_0, node.kwargs)
        kwargs = map_recursive(model_speedup.tensor_randomizer, kwargs)
        kwarg_masks = map_recursive(model_speedup.slot_getter_mask_1, node.kwargs)
        kwargs = map_recursive_zip(model_speedup.mask_applier, kwargs, kwarg_masks)
        map_recursive(model_speedup.tensor_requires_grad, kwargs)

        output = getattr(model_speedup, node.op)(node.target, args, kwargs)
        
        map_recursive_zip(model_speedup.indirect_update_param_mask, output, output_masks_2)

        # if node.op == 'call_module':
        #     # update the sparsity of the paramters
        #     node_info: NodeInfo = model_speedup.node_infos[node]
        #     sub_module: nn.Module = model_speedup.fetch_attr(node.target)
        #     for k, v in sub_module.named_parameters():
        #         grad_zero = v.grad.data == 0
        #         node_info.param_masks_1[k] = node_info.param_masks_0[k].clone()
        #         node_info.param_masks_1[k][grad_zero] = 0


        arg_values_2 = map_recursive(model_speedup.slot_getter_value_2, node.args)
        kwarg_values_2 = map_recursive(model_speedup.slot_getter_value_2, node.kwargs)

        map_recursive_zip(model_speedup.indirect_pass_grad, arg_values_2, args)
        map_recursive_zip(model_speedup.indirect_pass_grad, kwarg_values_2, kwargs)

    def indirect_update_postprocess(self, model_speedup: 'ModelSpeedup', node: Node) -> bool:
        pass


class DefaultModuleMaskUpdater(DefaultMaskUpdater):

    def detect(self, model_speedup: 'ModelSpeedup', node: Node) -> bool:
        """
        the default MaskUpdater for module, so return true if the node is a module calling
        """
        if node.op == 'call_module':
            model_speedup.node_infos[node].mask_updater = self
            sub_module: nn.Module = model_speedup.fetch_attr(node.target)
            param_masks = model_speedup.masks_file.get(node.target, {})
            for k, v in sub_module.named_parameters():
                if k not in param_masks:
                    param_masks[k] = torch.ones_like(v)
            model_speedup.node_infos[node].param_masks_0 = param_masks
            return True
        else:
            return False

    def direct_update_preprocess(self, model_speedup: 'ModelSpeedup', node: Node) -> bool:
        super().direct_update_preprocess(model_speedup, node)
        with torch.no_grad():
            node_info: NodeInfo = model_speedup.node_infos[node]
            sub_module: nn.Module = model_speedup.fetch_attr(node.target)

            for _k, v in sub_module.named_parameters():
                randomize_tensor(v, model_speedup.randomize_range_float[0], model_speedup.randomize_range_float[1])

            for k, v in sub_module.named_parameters():
                v *= node_info.param_masks_0[k] # in-place addition

    def indirect_update_process(self, model_speedup: 'ModelSpeedup', node: Node) -> bool:
        super().indirect_update_process(model_speedup, node)
        # update the sparsity of the paramters
        node_info: NodeInfo = model_speedup.node_infos[node]
        sub_module: nn.Module = model_speedup.fetch_attr(node.target)
        for k, v in sub_module.named_parameters():
            grad_zero = v.grad.data == 0
            node_info.param_masks_1[k] = node_info.param_masks_0[k].clone()
            node_info.param_masks_1[k][grad_zero] = 0
