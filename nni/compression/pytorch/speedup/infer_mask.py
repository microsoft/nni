# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import numpy as np
import torch
import torch.nn as nn
from ..utils import randomize_tensor, torch_float_dtype, torch_integer_dtype
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

STD_DELTA = 1e-8


class AutoMaskInference:
    def __init__(self, module, dummy_input, in_masks=None, weight_mask=None, \
                output_mask=None, name=None, in_constants=None, state_dict=None, batch_dim=0):
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
        print(self.output.grad_fn)
        # exit()
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
        if isinstance(self.module, nn.Module):
            # the function should not has parameters
            # get all the parameter tensors of the target module
            for name, para in module.named_parameters():
                self.weights[name] = para
                if name not in self.weight_mask:
                    self.weight_mask[name] = torch.ones_like(para.data)
        self.name = name
        self.state_dict = state_dict
        # TODO support the other batch dimension in the future
        self.batch_dim = batch_dim

    def random_init(self, start=0.1, end=10):
        """
        Random initialize the weights of the module. The value of
        the tensor will not affect the mask auto inference.
        """
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
                    in_tensor.data = in_tensor.data * \
                        self.in_masks[tid] + \
                        (1-self.in_masks[tid]) * self.in_constants[tid]

                    # print(in_tensor.data)

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
            # print(mask_pos)
            # for bid in range(tout.size(0)):
            #     # Set the mask and constant for each the output tensor
            #     out_mask[bid][mask_pos] = 0
            #     constant[bid][mask_pos] = mean[mask_pos]
            out_mask[:, mask_pos] = 0
            constant[:, mask_pos] = mean[mask_pos]
        return out_mask, constant

    def clac_out_sparsity(self):
        """
        Calculate the output sparsity.
        """
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
            out = self.module(*self.dummy_input).clone().detach()
            if isinstance(out, torch.Tensor):
                out_mask, constant = self.isconstants(out)
            elif isinstance(out, tuple) or isinstance(out, list):
                out_mask = []
                constant = []
                for tout in out:
                    _mask, _constant = self.isconstants(tout)
                    out_mask.append(_mask)
                    constant.append(_constant)
            else:
                _logger.warn(
                    'Only support the OP whose output is tensor/tuple of tensor/list of tensor')

        # We also need random the parameters of the module, because if the weight of the model has
        # a unmasked 0, then our out sparsity inference may be wrong
        # However, after radomizing the weight/parameters, the constant in the output tensors may
        # be different from the constants that calculated from its original stata_dict. However,
        # so to get the right constant to eliminate the bias between model before and after sparsity
        # inference, we need to reload its state_dict and recalculate the constant

        # TODO can be optimized here, move the randomization at the begining
        if len(self.weights) > 0 and self.state_dict is not None:
            # print(self.module.weight)

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
                    _tmp[sparsity_pos] = t_out[sparsity_pos]
                    constant.append(_tmp)

        # print('%%%%%%%%%')
        # print(out_mask[0])
        # print(constant[0])
        # print('%%%%%%%%')
        # print(self.module(torch.zeros(1,3,3,3).cuda() ))
        # print(self.module.weight)
        # print(self.module.bias)
        # if self.name =='bn1':
        #     print(self.module.running_mean)
        #     print(self.module.running_var)
        #     exit(-1)

        return out_mask, constant
        # return out_mask

    def update_indirect_sparsity(self):
        """
        Find those hidden sparsity through gradient.
        """
        # if self.name == 'conv1':
            # print(self.output)
            # print(type(self.output))
            # print(self.output.grad)
            # exit()
        # Each node only update the output mask when we backwards
        # update the output mask
        if isinstance(self.output, torch.Tensor) and self.output.grad is not None:
            # if output have gradient which means this node has successor
            # nodes and the successor nodes have already update their indirect
            # sparsity
            # we can mask the values whose gradient is always zeros
            gradient_sum = torch.sum(torch.abs(self.output.grad.data), dim=0)
            # if self.name == 'conv1':
            #     print('Gradient Sum')
            #     print(gradient_sum)
                # exit()
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
        print(output.grad_fn)
        if output.grad_fn is None:
            # the output does not have the gradient function
            return
        # Note: output maybe tensor or list/tuple of tensors
        if isinstance(output, torch.Tensor):
            output.backward(self.output_mask)
        elif isinstance(output, list) or isinstance(output, tuple):
            for tid, t_out in enumerate(output):
                t_out.backward(self.output_mask[tid])
        # print('\n\nself.output')
        # print(self.output)
        # print('\n\nself.output_mask\n\n')
        # print(self.output_mask)
        # print('\n\nself.dummy_input\n\n')
        # print(self.dummy_input)
        # print('\n\nself.in_mask\n\n')
        # print(self.in_masks)
        # print('\n\noutput\n\n')
        # print(output)

        # print(self.weight)
        # update the sparsity of the paramters
        for para_name in self.weights:
            # print("!!!!!!!!!!!!")
            # print(para_name)
            # print(self.weights[para_name].grad)
            grad_zero = self.weights[para_name].grad.data == 0
            self.weight_mask[para_name][grad_zero] = 0
        # print(self.name)
        # for tin in self.dummy_input:
        #     print(tin.requires_grad)
        #     print(tin.grad)
        # exit()
        # print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

    def update_direct_sparsity(self):
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            out_sparsity, out_constant = self.clac_out_sparsity()
            if isinstance(out_sparsity, torch.Tensor):
                assert isinstance(self.output_mask, torch.Tensor)
                self.output_mask *= out_sparsity
            elif isinstance(out_sparsity, list):
                for i, _ in enumerate(out_sparsity):
                    self.output_mask[i] *= out_sparsity[i]
            else:
                _logger.warn('There is no output sparsity')
            # also save the out_constant
            self.out_constant = out_constant

    def unmask(self, t_unmask):
        """
        Unmask some values to resolve the conflict/interference between the masks.
        Note: the t_unmask indicates the values that should be unmasked in the output
        tensors. We work backwards to resolve the mask conflicts in the model. We can only
        infer the values need to be unmasked in the input tensor/parameters from the unmasked
        values in the output tensor.
        Parameters
        ---------
        t_unmask: torch.Tensor
            This tensor indicates the values that should be unmasked in the output tensor.
        Returns
        -------
        input_unmask: list
            The values in the input tensors that should be unmasked
        """
        pass  # unmask behavior is different when OPs is different
        # you need to konw the calculation logic in the model to unmask
        # the tensor

        # print('$$$$$$$$$$$$$$$')
        # print('UNMASK in', self.name)
        # # Enable the gradient
        # self.requires_grad_()
        # self.random_init()
        # self.apply_mask()
        # self.zero_grad()

        # # in case there is in_place operation in this node
        # tmp_dummy_input = [x.clone() if isinstance(
        #     x, torch.Tensor) else x for x in self.dummy_input]
        # output = self.module(*tmp_dummy_input)
        # # backwards to get the gradient
        # if isinstance(t_unmask, torch.Tensor):
        #     unmask_pos = t_unmask > 0
        #     print('Unmask position')
        #     print(torch.sum(unmask_pos, (0,2,3)))
        #     print('Output mask before unmask')
        #     print(torch.sum(self.output_mask, (0,2,3)))
        #     # use the unmask tensor as the gradient
        #     output.backward(t_unmask)
        #     # update the output mask
        #     self.output_mask[unmask_pos] = 1
        # elif isinstance(t_unmask, list) or isinstance(t_unmask, tuple):
        #     assert isinstance(output, list) or isinstance(output, tuple)
        #     # the length of unmask tensor list should be exactly same with t_unmask
        #     assert len(output) == len(t_unmask)
        #     for i, _ in enumerate(t_unmask):
        #         _unmask = t_unmask[i]
        #         unmask_pos = _unmask > 0
        #         _output = output[i]
        #         _output.backward(_unmask)
        #         self.output_mask[i][unmask_pos] = 1
        # # all the values whose gradient is larger that zero
        # # should be unmasked unmasked
        # # unmask the values in the parameters
        # for para_name in self.weights:
        #     gradient = self.weights[para_name].grad.data
        #     unmask_pos = gradient > 0
        #     self.weight_mask[para_name][unmask_pos] = 1
        # # check if there are values in the input tensors that should be unmasked
        # input_debug = []
        # input_unmask = []
        # for i, _ in enumerate(self.dummy_input):
        #     if not isinstance(self.dummy_input[i], torch.Tensor):
        #         continue
        #     gradient = self.dummy_input[i].grad.data
        #     unmask_pos = gradient > 0
        #     print('Unmask pos!!!!!!')
        #     print(torch.sum(unmask_pos,[0, 2, 3]))
        #     if torch.sum((unmask_pos.to(torch.float32) - self.in_masks[i]) > 0) > 0:
        #         # if there is a masked value need to be unmasked, 1 in the unmask_pos
        #         # and 0 in self.in_masks[i]
        #         self.in_masks[i][unmask_pos] = 1
        #         input_debug.append(self.input_debugname[i])
        #         input_unmask.append(unmask_pos.to(torch.float32))
        # print('\n\nNew output mask after unmasking')
        # print(torch.sum(self.output_mask, (0,2,3)))
        # print('input needto unmask')
        # print(input_unmask)
        # # print(torch.sum(input_unmask[0],(0,2,3)))
        # return input_debug, input_unmask
