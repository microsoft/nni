# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from types import MappingProxyType
import logging
from typing import Dict
import torch
import torch.nn as nn
from ..utils import randomize_tensor, torch_float_dtype, torch_integer_dtype

from nni.common.concrete_trace_utils.utils import run_onlyif_instance, map_aggregate_zip, map_aggregate

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

STD_DELTA = 1e-6

seeds = {
    'inter_var': None,
    'weight': None,
}

class AutoMaskInference:
    def __init__(self, out_debugname, module, speedup, dummy_input, weight_mask=None,
                 name=None, state_dict=None):
        """
        This class will infer the mask of the target module automatically.
        This update_direct_sparsity will infer the output mask according
        to the input masks, in constrast, update_indirect_sparsity will
        infer the input masks according to given output masks. The newly
        found sparsity will be incrementally updated to the original in_masks
        and output_mask.

        Parameters
        ----------
        module: torch.nn.Module/function
            The target module to infer the mask. Need to be callable.
        dummy_input: list of Tensor
            The dummy_input of the target module.
        speedup: ModelSpeedup
            The reference of the ModelSpeedup object.
        in_masks:  list of torch.Tensor
            The input masks of the target module, if in_masks is not None, then
            update_direct_sparsity and update_indirect_sparsity will incrementally
            update the given in_masks, else, AutoMaskInference will create a new
            in_masks for the target module.
        output_mask: torch.Tensor
            The output mask of the target module. Similar to in_masks, if output_mask
            is not None, then update_direct_sparsity and update_indirect_sparsity will
            incrementally update the given output_mask, else AutoMaskInference will create
            one output_mask for the target module.
        weight_mask: dict of the weight masks
            The weight masks of the target module, the key is the corresponding name of
            the mask. For example: {'weight':torch.ones(1000, 1000), bias:torch.ones(1000)}
        name: str
            Name of the target module.
        state_dict: dict of torch.Tensor
            The original values of the weights.

        """
        errmsg = '%s is not callable, should pass the nn.Module/function' % str(
            module)
        assert callable(module), errmsg
        self.module = module

        # Initialize the dummy_input
        assert isinstance(dummy_input, list)
        # if there are multiple input variables
        self.dummy_input = tuple(dummy_input)

        # Initialize the mask for output tensors
        self.orig_output = speedup.internal_result[out_debugname]

        if isinstance(module, nn.GroupNorm):
            self.out_masks = self.in_masks[0]
        else:
            if isinstance(self.orig_output, torch.Tensor):
                self.out_masks = torch.ones_like(self.orig_output)
            elif isinstance(self.orig_output, list) or isinstance(self.orig_output, tuple):
                self.out_masks = []
                for o_tensor in self.orig_output:
                    if isinstance(o_tensor, torch.Tensor):
                        self.out_masks.append(torch.ones_like(o_tensor))
                    else:
                        # if one of the outputs is not tensor, set the corresponding
                        # mask to None
                        self.out_masks.append(None)
            else:
                self.out_masks = None

        # Initialize the mask for the parameters
        self.weight_mask = {}
        if weight_mask:
            self.weight_mask.update(weight_mask)
        self.name = name
        if isinstance(self.module, nn.Module):
            # the function should not has parameters
            # get all the parameter tensors of the target module
            for name, para in module.named_parameters():
                if name not in self.weight_mask:
                    self.weight_mask[name] = torch.ones_like(para.data)
        
            self.saved_weights = MappingProxyType({
                k: v.clone() for k, v in module.named_parameters()
            })

        self.state_dict = state_dict
        # TODO support the other batch dimension in the future
        self.batch_dim = speedup.batch_dim
        self.batch_size = speedup.confidence

    def update_input_info(self, in_masks):
        # Initialize the masks for input tensors
        self.in_masks = in_masks
        for in_id, _ in enumerate(self.in_masks):
            if self.in_masks[in_id] is None and \
                    isinstance(self.dummy_input[in_id], torch.Tensor):
                # if the input mask is None then create a all-ones mask for corresponding input tensor
                self.in_masks[in_id] = torch.ones_like(self.dummy_input[in_id])
                # ones_like will put the created mask on the same device with the dummy_input

    def init_and_apply(self, start=0.1, end=8.0):
        input = self.randomize_input(self.dummy_input, start, end)
        input = self.apply_input_mask(input)

        if isinstance(self.module, nn.Module):
            self.randomize_weight(start, end)
            self.apply_weight_mask()
        return input

    def randomize_input(self, input, start, end):
        """
        Random initialize the weights of the module. The value of
        the tensor will not affect the mask auto inference.
        """
        # currently we set the random range to 0.1-8.0 because of the ReLU6,
        # if we use a range that far larger than 6, it may infer a wrong mask
        # when the confidence is low. In the future, we will add the mask inference
        # rules for ReLU6 to break this range constraint.
        with torch.no_grad():
            for tensor in input:
                if isinstance(tensor, torch.Tensor) and len(tensor.size()) > self.batch_dim\
                    and tensor.size(self.batch_dim) == self.batch_size:
                    # if the input tensor only has one dimension, which means
                    # it doesn't have the batch dimension, then we don't randomize
                    # this tensor, because our tensor scrambling is on the batch
                    # dimention. For example, if the tensor is a scalar(returned
                    # by the size operator), then we will skip this tensor
                    seeds['inter_var'] += 1
                    torch.manual_seed(seeds['inter_var'])
                    randomize_tensor(tensor, start, end)
                    torch.manual_seed(100)

        return input

    def randomize_weight(self, start, end):
        with torch.no_grad():
            for weight_key in self.saved_weights.keys():
                seeds['weight'] += 1
                torch.manual_seed(seeds['weight'])
                randomize_tensor(self.module.get_parameter(weight_key).data, start, end)
                torch.manual_seed(100)

    def zero_grad(self):
        """
        Set the gradient of the weight, input tensor to be zeros.
        """
        with torch.no_grad():
            # set the weight's gradient to zero
            if isinstance(self.module, nn.Module):
                self.module.zero_grad()
            # also zero the gradient of the input tensors
            for tensor in self.dummy_input:
                if isinstance(tensor, torch.Tensor):
                    if tensor.grad is not None:
                        tensor.grad.data.zero_()

    def requires_grad_(self, dummy_input, flag=True):
        """
        Set the requires_grad of input tensor and parameters to flag.
        """
        for t_in in dummy_input:
            if isinstance(t_in, torch.Tensor) and t_in.dtype in torch_float_dtype:
                # only float type can require the gradient
                # enable the auto gradient
                t_in.requires_grad_(flag)

        if isinstance(self.module, nn.Module):
            for weight_key in self.saved_weights:
                if self.module.get_parameter(weight_key).dtype in torch_float_dtype:
                    self.module.get_parameter(weight_key).requires_grad_(flag)

    def apply_input_mask(self, input):
        """
        Apply the mask of the input tensor.
        """
        with torch.no_grad():
            # apply the input mask
            for tid, in_tensor in enumerate(input):
                if isinstance(in_tensor, torch.Tensor) and self.in_masks[tid] is not None:
                    # issue-4540 when two tensors are multiplied, the constants part make
                    # the propagation weaker, and lead to shape misaligment. Currently, we
                    # donnot support the constant folding, so, we just remove the constant here
                    in_tensor.data = in_tensor.data * self.in_masks[tid]
                    # in_tensor.data *= self.in_masks[tid]
                    # in_tensor *= self.in_masks[tid]
        return input

    def apply_weight_mask(self):
        """
        Apply the weight mask of this module.
        """
        with torch.no_grad():
            # apply the weight mask
            for weight_key in self.saved_weights.keys():
                if weight_key in self.weight_mask:
                    self.module.register_parameter(
                        weight_key,
                        torch.nn.Parameter(self.module.get_parameter(weight_key) * self.weight_mask[weight_key])
                    )

    def calc_out_masks(self, out):
        if isinstance(out, torch.Tensor):
            out_mask = self.calc_one_masked_mask(out.clone().detach())
        elif isinstance(out, tuple) or isinstance(out, list):
            out_mask = [self.calc_one_masked_mask(tout.clone().detach()) for tout in out]
        else:
            _logger.warning(
                'Only support the OP whose output is tensor/tuple of tensor/list of tensor')
            out_mask = None

        # We also need random the parameters of the module, because if the weight of the model has
        # a unmasked 0, then our out sparsity inference may be wrong
        # However, after radomizing the weight/parameters, the constant in the output tensors may
        # be different from the constants that calculated from its original stata_dict. However,
        # so to get the right constant to eliminate the bias between model before and after sparsity
        # inference, we need to reload its state_dict and recalculate the constant
        # Currently we also get the constant values at the same time when infering the mask, in
        # the future, we will separate the constant inference into a single graph pass.
        if isinstance(out_mask, torch.Tensor):
            assert isinstance(self.out_masks, torch.Tensor)
            self.out_masks *= out_mask
        elif isinstance(out_mask, list):
            for i, _ in enumerate(out_mask):
                self.out_masks[i] *= out_mask[i]
        else:
            _logger.warning('There is no output sparsity')

    def calc_one_masked_mask(self, tout):
        """
        Find the constants in the tensor tout. This function return a mask tensor that
        indicates if a value in tout is a constant, and return one more tensor to indicate
        that the values of the constant.

        Paramters
        ---------
        tout: torch.Tensor
            The target output tensor to find the constants
        Returns
        -------
        mask: torch.Tensor
            The mask tensor(same shape with tout) that indicates that whether
            the correponding value is a constant.
        """
        assert isinstance(tout, torch.Tensor)
        out_mask = torch.ones_like(tout)
        # judge if tout is a scalar(tensor that only have one value)
        if len(tout.size()) == 0:
            # tout is a scalar tensor, for the scalar tensor, we take
            # this scalar as a constant, usually, the scalar tensor is returned
            # by the size() function
            return out_mask
        if tout.dtype in torch_integer_dtype:
            # Pytorch cannot use torch.mean and torch.std to process
            # intergers :( , so if dtype of the input tensor is integer, we need
            # check if is the constant by ourselves
            # Note: the first dimension should be the batch dimension
            same = tout[:] == tout[0]
            reduced = torch.sum(same, dim=0)
            is_constant = reduced == tout.size(0)
            out_mask[:, is_constant] = 0

        else:
            # calculate the std of the output among batch dimension
            std = torch.std(tout, dim=0)
            mask_pos = std < STD_DELTA
            out_mask[:, mask_pos] = 0
        return out_mask

    def update_indirect_out_mask(self):
        # Each node only update the output mask when we backwards
        # update the output mask, this is because that some op may
        # have the broadcast operation, for example, OP A's output
        # tensor may be taken by two OPs(B, C) as inputs. So we cannot
        # directly update the input mask at the OP B or C. We can only
        # update the mask of C's output tensor only when B and C are
        # already updated(gradient are already calculated and added to
        # C's output tensor).
        # Besides, updating the mask of C's output tensor equals to updating
        # the input mask of OP B and C.
        if isinstance(self.orig_output, torch.Tensor) and self.orig_output.grad is not None:
            # if output have gradient which means this node has successor
            # nodes and the successor nodes have already update their indirect
            # sparsity
            # we can mask the values whose gradient is always zeros
            gradient_sum = torch.sum(torch.abs(self.orig_output.grad.data), dim=0)
            _grad_zero = gradient_sum == 0
            for batchid in range(self.orig_output.size(0)):
                # set the same mask value for the whole batche
                self.out_masks[batchid][_grad_zero] = 0
        elif isinstance(self.orig_output, tuple) or isinstance(self.orig_output, list):
            assert isinstance(self.out_masks, (tuple, list))
            for oid, tout in enumerate(self.orig_output):
                errmsg = 'The output only support tensor/list of tensors'
                assert isinstance(tout, torch.Tensor), errmsg
                gradient_sum = torch.sum(
                    torch.abs(self.orig_output.grad.data), dim=0)
                _grad_zero = gradient_sum == 0
                for batchid in range(self.orig_output.size(0)):
                    # set the same mask value for the whole batch
                    self.out_masks[oid][batchid][_grad_zero] = 0

    def update_indirect_weight_mask_helper(self, output, out_mask):
        # Note: output maybe tensor or list/tuple of tensors
        if isinstance(output, torch.Tensor):
            assert isinstance(out_mask, torch.Tensor)
            if output.grad_fn is not None:
                output.backward(out_mask)
        else:
            assert not isinstance(out_mask, torch.Tensor)

    def update_indirect_weight_mask(self, output):
        map_aggregate_zip(self.update_indirect_weight_mask_helper, output, self.out_masks)

        # update the sparsity of the paramters
        if isinstance(self.module, nn.Module):
            for weight_key in self.saved_weights.keys():
                grad_zero = self.module.get_parameter(weight_key).grad.data == 0
                self.weight_mask[weight_key][grad_zero] = 0

    def update_indirect_sparsity(self):
        """
        This function will update the indirect sparsity. To explain what's
        indirect sparsity, for example, there is two tensors TA and TB, and
        we perform the calculation: TC = TA x TB in which TC is also a tensor.
        Once some values in TA are masked to zeros, then the corresponding
        positions in TB are also potential sparsities, because these have no
        effect of the final output(the gradient of these positions in TB equal
        to 0 all the time). This function it to fine the potential sparsity caused
        by other sparsity(we call it indirect sparsity here). Basically we can find
        these potential sparsity through gradient.
        """
        
        self.update_indirect_out_mask()

        # Forward inference with auto gradient enabled
        # Note: tensors that need gradient cannot be used in the in-place operator
        the_dummy_input = self.init_and_apply()
        self.requires_grad_(the_dummy_input, True)
        # Some operator may have the in_place operations, so we need to clone the input
        # before passing to the self.module
        the_dummy_input = [x.clone() for x in the_dummy_input]
        output = self.module(*the_dummy_input)
        self.update_indirect_weight_mask(output)

    def update_direct_sparsity(self):
        # we don't need the gradient in the forward inference
        out_mask = None
        with torch.no_grad():
            the_dummy_input = self.init_and_apply()
            out = self.module(*the_dummy_input)
            self.calc_out_masks(out)

    def get_masks(self):
        return (self.in_masks, self.out_masks, self.weight_mask)
