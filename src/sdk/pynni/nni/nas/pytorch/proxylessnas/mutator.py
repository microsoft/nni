# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge,
# to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from nni.nas.pytorch.base_mutator import BaseMutator

def detach_variable(inputs):
    if isinstance(inputs, tuple):
        return tuple([detach_variable(x) for x in inputs])
    else:
        x = inputs.detach()
        x.requires_grad = inputs.requires_grad
        return x

class ArchGradientFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, binary_gates, run_func, backward_func):
        ctx.run_func = run_func
        ctx.backward_func = backward_func

        detached_x = detach_variable(x)
        with torch.enable_grad():
            output = run_func(detached_x)
        ctx.save_for_backward(detached_x, output)
        return output.data

    @staticmethod
    def backward(ctx, grad_output):
        detached_x, output = ctx.saved_tensors

        grad_x = torch.autograd.grad(output, detached_x, grad_output, only_inputs=True)
        # compute gradients w.r.t. binary_gates
        binary_grads = ctx.backward_func(detached_x.data, output.data, grad_output.data)

        return grad_x[0], binary_grads, None, None

class MixedOp(nn.Module):
    """
    This class is to instantiate and manage info of one LayerChoice
    """
    def __init__(self, mutable):
        """
        Parameters
        ----------
        mutable : LayerChoice
            A LayerChoice in user model
        """
        super(MixedOp, self).__init__()
        self.mutable = mutable
        self.AP_path_alpha = nn.Parameter(torch.Tensor(mutable.length))
        self.AP_path_wb = nn.Parameter(torch.Tensor(mutable.length))
        self.active_index = [0]
        self.inactive_index = None
        self.log_prob = None
        self.current_prob_over_ops = None
        self.n_choices = mutable.length

    def get_AP_path_alpha(self):
        return self.AP_path_alpha

    def forward(self, mutable, x):
        # only full_v2
        def run_function(key, candidate_ops, active_id):
            def forward(_x):
                return candidate_ops[active_id](_x)
            return forward

        def backward_function(key, candidate_ops, active_id, binary_gates):
            def backward(_x, _output, grad_output):
                binary_grads = torch.zeros_like(binary_gates.data)
                with torch.no_grad():
                    for k in range(len(candidate_ops)):
                        if k != active_id:
                            out_k = candidate_ops[k](_x.data)
                        else:
                            out_k = _output.data
                        grad_k = torch.sum(out_k * grad_output)
                        binary_grads[k] = grad_k
                return binary_grads
            return backward
        output = ArchGradientFunction.apply(
            x, self.AP_path_wb, run_function(mutable.key, mutable.choices, self.active_index[0]),
            backward_function(mutable.key, mutable.choices, self.active_index[0], self.AP_path_wb))
        return output

    @property
    def probs_over_ops(self):
        """
        Apply softmax on alpha to generate probability distribution

        Returns
        -------
        pytorch tensor
            probability distribution
        """
        probs = F.softmax(self.AP_path_alpha, dim=0)  # softmax to probability
        return probs

    @property
    def chosen_index(self):
        """ choose the max one """
        probs = self.probs_over_ops.data.cpu().numpy()
        index = int(np.argmax(probs))
        return index, probs[index]

    @property
    def active_op(self):
        """ assume only one path is active """
        return self.mutable.choices[self.active_index[0]]

    @property
    def active_op_index(self):
        """ return active op's index """
        return self.active_index[0]

    def set_chosen_op_active(self):
        """ set chosen index, active and inactive indexes """
        chosen_idx, _ = self.chosen_index
        self.active_index = [chosen_idx]
        self.inactive_index = [_i for _i in range(0, chosen_idx)] + \
                              [_i for _i in range(chosen_idx + 1, self.n_choices)]

    def binarize(self):
        """
        Sample based on alpha, and set binary weights accordingly
        """
        self.log_prob = None
        # reset binary gates
        self.AP_path_wb.data.zero_()
        probs = self.probs_over_ops
        sample = torch.multinomial(probs, 1)[0].item()
        self.active_index = [sample]
        self.inactive_index = [_i for _i in range(0, sample)] + \
                              [_i for _i in range(sample + 1, len(self.mutable.choices))]
        self.log_prob = torch.log(probs[sample])
        self.current_prob_over_ops = probs
        self.AP_path_wb.data[sample] = 1.0
        # avoid over-regularization
        for choice in self.mutable.choices:
            for _, param in choice.named_parameters():
                param.grad = None

    def _delta_ij(self, i, j):
        if i == j:
            return 1
        else:
            return 0

    def set_arch_param_grad(self):
        """
        Calculate alpha gradient for this LayerChoice
        """
        binary_grads = self.AP_path_wb.grad.data
        if self.active_op.is_zero_layer():
            self.AP_path_alpha.grad = None
            return
        if self.AP_path_alpha.grad is None:
            self.AP_path_alpha.grad = torch.zeros_like(self.AP_path_alpha.data)
        probs = self.probs_over_ops.data
        for i in range(len(self.mutable.choices)):
            for j in range(len(self.mutable.choices)):
                self.AP_path_alpha.grad.data[i] += binary_grads[j] * probs[j] * (self._delta_ij(i, j) - probs[i])


class ProxylessNasMutator(BaseMutator):
    def __init__(self, model):
        """
        Init a MixedOp instance for each named mutable i.e., LayerChoice

        Parameters
        ----------
        model : pytorch model
            The model that users want to tune, it includes search space defined with nni nas apis
        """
        super(ProxylessNasMutator, self).__init__(model)
        self.mixed_ops = {}
        for _, mutable, _ in self.named_mutables(distinct=False):
            self.mixed_ops[mutable.key] = MixedOp(mutable)

    def on_forward_layer_choice(self, mutable, *inputs):
        """
        Callback of layer choice forward. Override if you are an advanced user.
        On default, this method calls :meth:`on_calc_layer_choice_mask` to get a mask on how to choose between layers
        (either by switch or by weights), then it will reduce the list of all tensor outputs with the policy speicified
        in `mutable.reduction`. It will also cache the mask with corresponding `mutable.key`.

        Parameters
        ----------
        mutable: LayerChoice
        inputs: list of torch.Tensor

        Returns
        -------
        torch.Tensor
        index of the chosen op
        """
        # FIXME: return mask, to be consistent with other algorithms
        idx = self.mixed_ops[mutable.key].active_op_index
        return self.mixed_ops[mutable.key](mutable, *inputs), idx

    def reset_binary_gates(self):
        """
        For each LayerChoice, binarize based on alpha to only activate one op
        """
        for k in self.mixed_ops.keys():
            self.mixed_ops[k].binarize()

    def set_chosen_op_active(self):
        """
        For each LayerChoice, set the op with highest alpha as the chosen op
        Usually used for validation.
        """
        for k in self.mixed_ops.keys():
            self.mixed_ops[k].set_chosen_op_active()

    def num_arch_params(self):
        """
        Returns
        -------
        The number of LayerChoice in user model
        """
        return len(self.mixed_ops)

    def set_arch_param_grad(self):
        """
        For each LayerChoice, calculate gradients for architecture weights, i.e., alpha
        """
        for k in self.mixed_ops.keys():
            self.mixed_ops[k].set_arch_param_grad()

    def get_architecture_parameters(self):
        """
        Return architecture weights of each LayerChoice, for arch optimizer
        """
        for k in self.mixed_ops.keys():
            yield self.mixed_ops[k].get_AP_path_alpha()
