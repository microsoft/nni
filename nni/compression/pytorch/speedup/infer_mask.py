# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import torch
import torch.nn as nn
from ..utils import randomize_tensor, torch_float_dtype, torch_integer_dtype
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

STD_DELTA = 1e-6


class AutoMaskInference:
    def __init__(self, module, dummy_input, in_masks=None, weight_mask=None, \
                output_mask=None, name=None, in_constants=None, state_dict=None, batch_dim=0):
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
        dummy_input: torch.Tensor/list of Tensor
            The dummy_input of the target module.
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
        in_constants: list of torch.Tensor
            The correponding constant values of the in_masks.
        state_dict: dict of torch.Tensor
            The original values of the weights.
        batch_dim: int
            The index of the batch dimension of the input tensors.

        """
        errmsg = '%s is not callable, should pass the nn.Module/function' % str(
            module)
        assert callable(module), errmsg
        self.module = module

        # Initialize the dummy_input
        if isinstance(dummy_input, list):
            # if there are multiple input variables
            self.dummy_input = dummy_input
        else:
            # if there is only one input variable
            self.dummy_input = [dummy_input]

        # Initialize the masks for input tensors
        self.in_masks = in_masks if in_masks is not None else [
            None] * len(self.dummy_input)
        self.in_constants = in_constants if in_constants is not None else [
            torch.zeros_like(x) for x in dummy_input]
        for in_id, _ in enumerate(self.in_masks):
            if self.in_masks[in_id] is None and \
                    isinstance(self.dummy_input[in_id], torch.Tensor):
                # if the input mask is None then create a all-ones mask for corresponding input tensor
                self.in_masks[in_id] = torch.ones_like(self.dummy_input[in_id])
                # ones_like will put the created mask on the same device with the dummy_input

        # Initialize the mask for output tensors
        self.output = self.module(*dummy_input)
        # self.output.requires_grad_()
        if output_mask is not None:
            # assume the given output mask is right
            self.output_mask = output_mask
        else:
            if isinstance(self.output, torch.Tensor):
                self.output_mask = torch.ones_like(self.output)
            elif isinstance(self.output, list) or isinstance(self.output, tuple):
                self.output_mask = []
                for o_tensor in self.output:
                    if isinstance(o_tensor, torch.Tensor):
                        self.output_mask.append(torch.ones_like(o_tensor))
                    else:
                        # if one of the outputs is not tensor, set the corresponding
                        # mask to None
                        self.output_mask.append(None)
            else:
                self.output_mask = None

        # Initialize the mask for the parameters
        self.weights = {}
        self.weight_mask = {}
        if weight_mask:
            self.weight_mask.update(weight_mask)
        self.name = name
        if isinstance(self.module, nn.Module):
            # the function should not has parameters
            # get all the parameter tensors of the target module
            for name, para in module.named_parameters():
                self.weights[name] = para
                if name not in self.weight_mask:
                    self.weight_mask[name] = torch.ones_like(para.data)
        self.state_dict = state_dict
        # TODO support the other batch dimension in the future
        self.batch_dim = batch_dim

    def random_init(self, start=0.1, end=8.0):
        """
        Random initialize the weights of the module. The value of
        the tensor will not affect the mask auto inference.
        """
        # currently we set the random range to 0.1-8.0 because of the ReLU6,
        # if we use a range that far larger than 6, it may infer a wrong mask
        # when the confidence is low. In the future, we will add the mask inference
        # rules for ReLU6 to break this range constraint.
        with torch.no_grad():
            for tensor in self.dummy_input:
                if isinstance(tensor, torch.Tensor) and len(tensor.size()) > 0:
                    # if the tensor is a scalar, then skip this tensor
                    randomize_tensor(tensor, start, end)
            for para in self.weights:
                randomize_tensor(self.weights[para].data, start, end)


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

    def requires_grad_(self, flag=True):
        """
        Set the requires_grad of input tensor and parameters to flag.
        """
        for t_in in self.dummy_input:
            if isinstance(t_in, torch.Tensor) and t_in.dtype in torch_float_dtype:
                # only float type can require the gradient
                # enable the auto gradient
                t_in.requires_grad_(flag)
        for para_name in self.weights:
            if self.weights[para_name].dtype in torch_float_dtype:
                self.weights[para_name].requires_grad_(flag)

    def apply_mask(self):
        self.__apply_input_mask()
        self.__apply_weight_mask()

    def __apply_input_mask(self):
        """
        Apply the mask of the input tensor.
        """
        with torch.no_grad():
            # apply the input mask
            for tid, in_tensor in enumerate(self.dummy_input):
                if isinstance(in_tensor, torch.Tensor) and self.in_masks[tid] is not None:
                    # in_tensor.data = in_tensor.data * \
                    #     self.in_masks[tid] + \
                    #     (1-self.in_masks[tid]) * self.in_constants[tid]
                    # issue-4540 when two tensors are multiplied, the constants part make
                    # the propagation weaker, and lead to shape misaligment. Currently, we
                    # donnot support the constant folding, so, we just remove the constant here
                    in_tensor.data = in_tensor.data * \
                        self.in_masks[tid]

    def __apply_weight_mask(self):
        """
        Apply the weight mask of this module.
        """
        with torch.no_grad():
            # apply the weight mask
            for para in self.weights:
                if para in self.weight_mask:
                    self.weights[para].data *= self.weight_mask[para].data

    def isconstants(self, tout):
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
        constant: torch.Tensor
            The mask tensot(same shape with tout) that indicates the values of
            the constants in the tout.
        """
        assert isinstance(tout, torch.Tensor)
        out_mask = torch.ones_like(tout)
        constant = torch.zeros_like(tout)
        # judge if tout is a scalar(tensor that only have one value)
        if len(tout.size()) == 0:
            # tout is a scalar tensor, for the scalar tensor, we take
            # this scalar as a constant, usually, the scalar tensor is returned
            # by the size() function
            constant = tout
            return out_mask, constant
        if tout.dtype in torch_integer_dtype:
            # Pytorch cannot use torch.mean and torch.std to process
            # intergers :( , so if dtype of the input tensor is integer, we need
            # check if is the constant by ourselves
            # Note: the first dimension should be the batch dimension
            same = tout[:] == tout[0]
            reduced = torch.sum(same, dim=0)
            is_constant = reduced == tout.size(0)
            out_mask[:, is_constant] = 0
            constant[:, is_constant] = tout[0][is_constant]

        else:
            # calculate the std of the output among batch dimension
            std = torch.std(tout, dim=0)
            # calculate the mean value of the output among the batch dimension
            mean = torch.mean(tout, dim=0)
            mask_pos = std < STD_DELTA
            out_mask[:, mask_pos] = 0
            constant[:, mask_pos] = mean[mask_pos]
        return out_mask, constant


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
        if isinstance(self.output, torch.Tensor) and self.output.grad is not None:
            # if output have gradient which means this node has successor
            # nodes and the successor nodes have already update their indirect
            # sparsity
            # we can mask the values whose gradient is always zeros
            gradient_sum = torch.sum(torch.abs(self.output.grad.data), dim=0)
            _grad_zero = gradient_sum == 0
            for batchid in range(self.output.size(0)):
                # set the same mask value for the whole batche
                self.output_mask[batchid][_grad_zero] = 0
        elif isinstance(self.output, tuple) or isinstance(self.output, list):
            assert isinstance(self.output_mask, (tuple, list))
            for oid, tout in enumerate(self.output):
                errmsg = 'The output only support tensor/list of tensors'
                assert isinstance(tout, torch.Tensor), errmsg
                gradient_sum = torch.sum(
                    torch.abs(self.output.grad.data), dim=0)
                _grad_zero = gradient_sum == 0
                for batchid in range(self.output.size(0)):
                    # set the same mask value for the whole batch
                    self.output_mask[oid][batchid][_grad_zero] = 0

        self.requires_grad_(True)
        # Forward inference with auto gradient enabled
        # Note: tensors that need gradient cannot be used in the in-place operator
        self.random_init()
        self.apply_mask()
        # Some operator may have the in_place operations, so we need to clone the input
        # before passing to the self.module
        tmp_dummy_input = [x.clone() if isinstance(
            x, torch.Tensor) else x for x in self.dummy_input]
        output = self.module(*tmp_dummy_input)

        if output.grad_fn is None:
            # the output does not have the gradient function
            return
        # Note: output maybe tensor or list/tuple of tensors
        if isinstance(output, torch.Tensor):
            output.backward(self.output_mask)
        elif isinstance(output, list) or isinstance(output, tuple):
            for tid, t_out in enumerate(output):
                t_out.backward(self.output_mask[tid])

        # update the sparsity of the paramters
        for para_name in self.weights:
            grad_zero = self.weights[para_name].grad.data == 0
            self.weight_mask[para_name][grad_zero] = 0

    def update_direct_sparsity(self):
        # we don't need the gradient in the forward inference
        out_mask = None
        constant = None
        with torch.no_grad():
            # Note: we need randomly init the input one more time here!
            # Because some operation have the in-place operation, such as relu_,
            # the in-place operation may modify or write 0s into the dummy_input
            self.random_init()
            # apply the mask for the input tensor and the weight tensor
            self.apply_mask()
            # Note: due to the in-place operator, such as relu_,
            # ori_out may be the same tensor with dummy_input,
            # so we use clone and detach to create a new tensor with
            # the same values.
            out = self.module(*self.dummy_input)
            if isinstance(out, torch.Tensor):
                out_mask, constant = self.isconstants(out.clone().detach())
            elif isinstance(out, tuple) or isinstance(out, list):
                out_mask = []
                constant = []
                for tout in out:
                    _mask, _constant = self.isconstants(tout.clone().detach())
                    out_mask.append(_mask)
                    constant.append(_constant)
            else:
                _logger.warning(
                    'Only support the OP whose output is tensor/tuple of tensor/list of tensor')

            # We also need random the parameters of the module, because if the weight of the model has
            # a unmasked 0, then our out sparsity inference may be wrong
            # However, after radomizing the weight/parameters, the constant in the output tensors may
            # be different from the constants that calculated from its original stata_dict. However,
            # so to get the right constant to eliminate the bias between model before and after sparsity
            # inference, we need to reload its state_dict and recalculate the constant
            # Currently we also get the constant values at the same time when infering the mask, in
            # the future, we will separate the constant inference into a single graph pass.
            if len(self.weights) > 0 and self.state_dict is not None:

                self.module.load_state_dict(self.state_dict)
                # apply weight mask
                self.__apply_weight_mask()
                out = self.module(*self.dummy_input).clone().detach()
                if isinstance(out, torch.Tensor):
                    constant = torch.zeros_like(out)
                    constant_pos = out_mask == 0
                    constant[constant_pos] = out[constant_pos]
                elif isinstance(out, (list, tuple)):
                    constant = []
                    for i, tout in enumerate(out):
                        _tmp = torch.zeros_like(tout)
                        sparsity_pos = out_mask[i] == 0
                        _tmp[sparsity_pos] = tout[sparsity_pos]
                        constant.append(_tmp)

            if isinstance(out_mask, torch.Tensor):
                assert isinstance(self.output_mask, torch.Tensor)
                self.output_mask *= out_mask
            elif isinstance(out_mask, list):
                for i, _ in enumerate(out_mask):
                    self.output_mask[i] *= out_mask[i]
            else:
                _logger.warning('There is no output sparsity')
            # also save the out_constant
            self.out_constant = constant

    def get_masks(self):
        return (self.in_masks, self.output_mask, self.weight_mask)

