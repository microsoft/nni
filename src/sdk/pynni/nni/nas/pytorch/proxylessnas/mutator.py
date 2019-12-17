# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from .utils import detach_variable
from nni.nas.pytorch.base_mutator import BaseMutator
from nni.nas.pytorch.mutables import LayerChoice

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
    forward_mode = None
    def __init__(self, mutable):
        """
        Parameters
        ----------
        mutable : LayerChoice
            A LayerChoice in user model
        """
        super(MixedOp, self).__init__()
        self.AP_path_alpha = nn.Parameter(torch.Tensor(mutable.length))
        self.AP_path_wb = nn.Parameter(torch.Tensor(mutable.length))
        self.AP_path_alpha.requires_grad = False
        self.AP_path_wb.requires_grad = False
        self.active_index = [0]
        self.inactive_index = None
        self.log_prob = None
        self.current_prob_over_ops = None
        self.n_choices = mutable.length

    def get_AP_path_alpha(self):
        return self.AP_path_alpha

    def to_requires_grad(self):
        self.AP_path_alpha.requires_grad = True
        self.AP_path_wb.requires_grad = True

    def disable_grad(self):
        self.AP_path_alpha.requires_grad = False
        self.AP_path_wb.requires_grad = False

    def forward(self, mutable, x):
        if MixedOp.forward_mode == 'full' or MixedOp.forward_mode == 'two':
            output = 0
            for _i in self.active_index:
                oi = self.candidate_ops[_i](x)
                output = output + self.AP_path_wb[_i] * oi
            for _i in self.inactive_index:
                oi = self.candidate_ops[_i](x)
                output = output + self.AP_path_wb[_i] * oi.detach()
        elif MixedOp.forward_mode == 'full_v2':
            # does not work in DataParallel, possible memory leak
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
        else:
            output = self.active_op(mutable)(x)
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

    def active_op(self, mutable):
        """ assume only one path is active """
        return mutable.choices[self.active_index[0]]

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

    def binarize(self, mutable):
        """
        Sample based on alpha, and set binary weights accordingly
        """
        self.log_prob = None
        # reset binary gates
        self.AP_path_wb.data.zero_()
        probs = self.probs_over_ops
        if MixedOp.forward_mode == 'two':
            # sample two ops according to probs
            sample_op = torch.multinomial(probs.data, 2, replacement=False)
            probs_slice = F.softmax(torch.stack([
                self.AP_path_alpha[idx] for idx in sample_op
            ]), dim=0)
            self.current_prob_over_ops = torch.zeros_like(probs)
            for i, idx in enumerate(sample_op):
                self.current_prob_over_ops[idx] = probs_slice[i]
            # choose one to be active and the other to be inactive according to probs_slice
            c = torch.multinomial(probs_slice.data, 1)[0] # 0 or 1
            active_op = sample_op[c].item()
            inactive_op = sample_op[1-c].item()
            self.active_index = [active_op]
            self.inactive_index = [inactive_op]
            # set binary gate
            self.AP_path_wb.data[active_op] = 1.0
        else:
            sample = torch.multinomial(probs, 1)[0].item()
            self.active_index = [sample]
            self.inactive_index = [_i for _i in range(0, sample)] + \
                                [_i for _i in range(sample + 1, len(mutable.choices))]
            self.log_prob = torch.log(probs[sample])
            self.current_prob_over_ops = probs
            self.AP_path_wb.data[sample] = 1.0
        # avoid over-regularization
        for choice in mutable.choices:
            for _, param in choice.named_parameters():
                param.grad = None

    def _delta_ij(self, i, j):
        if i == j:
            return 1
        else:
            return 0

    def set_arch_param_grad(self, mutable):
        """
        Calculate alpha gradient for this LayerChoice
        """
        binary_grads = self.AP_path_wb.grad.data
        if self.active_op(mutable).is_zero_layer():
            self.AP_path_alpha.grad = None
            return
        if self.AP_path_alpha.grad is None:
            self.AP_path_alpha.grad = torch.zeros_like(self.AP_path_alpha.data)
        if MixedOp.forward_mode == 'two':
            involved_idx = self.active_index + self.inactive_index
            probs_slice = F.softmax(torch.stack([
                self.AP_path_alpha[idx] for idx in involved_idx
            ]), dim=0).data
            for i in range(2):
                for j in range(2):
                    origin_i = involved_idx[i]
                    origin_j = involved_idx[j]
                    self.AP_path_alpha.grad.data[origin_i] += \
                        binary_grads[origin_j] * probs_slice[j] * (self._delta_ij(i, j) - probs_slice[i])
            for _i, idx in enumerate(self.active_index):
                self.active_index[_i] = (idx, self.AP_path_alpha.data[idx].item())
            for _i, idx in enumerate(self.inactive_index):
                self.inactive_index[_i] = (idx, self.AP_path_alpha.data[idx].item())
        else:
            probs = self.probs_over_ops.data
            for i in range(self.n_choices):
                for j in range(self.n_choices):
                    self.AP_path_alpha.grad.data[i] += binary_grads[j] * probs[j] * (self._delta_ij(i, j) - probs[i])
        return

    def rescale_updated_arch_param(self):
        if not isinstance(self.active_index[0], tuple):
            assert self.active_op.is_zero_layer()
            return
        involved_idx = [idx for idx, _ in (self.active_index + self.inactive_index)]
        old_alphas = [alpha for _, alpha in (self.active_index + self.inactive_index)]
        new_alphas = [self.AP_path_alpha.data[idx] for idx in involved_idx]

        offset = math.log(
            sum([math.exp(alpha) for alpha in new_alphas]) / sum([math.exp(alpha) for alpha in old_alphas])
        )

        for idx in involved_idx:
            self.AP_path_alpha.data[idx] -= offset


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
        self._unused_modules = None
        self.mutable_list = []
        for mutable in self.undedup_mutables:
            mo = MixedOp(mutable)
            self.mutable_list.append(mutable)
            mutable.registered_module = mo

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
        idx = mutable.registered_module.active_op_index
        return mutable.registered_module(mutable, *inputs), idx

    def reset_binary_gates(self):
        """
        For each LayerChoice, binarize based on alpha to only activate one op
        """
        for mutable in self.undedup_mutables:
            mutable.registered_module.binarize(mutable)

    def set_chosen_op_active(self):
        """
        For each LayerChoice, set the op with highest alpha as the chosen op
        Usually used for validation.
        """
        for mutable in self.undedup_mutables:
            mutable.registered_module.set_chosen_op_active()

    def num_arch_params(self):
        """
        Returns
        -------
        The number of LayerChoice in user model
        """
        return len(self.mutable_list)

    def set_arch_param_grad(self):
        """
        For each LayerChoice, calculate gradients for architecture weights, i.e., alpha
        """
        for mutable in self.undedup_mutables:
            mutable.registered_module.set_arch_param_grad(mutable)

    def get_architecture_parameters(self):
        """
        Return architecture weights of each LayerChoice, for arch optimizer
        """
        for mutable in self.undedup_mutables:
            yield mutable.registered_module.get_AP_path_alpha()

    def change_forward_mode(self, mode):
        MixedOp.forward_mode = mode

    def get_forward_mode(self):
        return MixedOp.forward_mode

    def rescale_updated_arch_param(self):
        for mutable in self.undedup_mutables:
            mutable.registered_module.rescale_updated_arch_param()

    def unused_modules_off(self):
        self._unused_modules = []
        for mutable in self.undedup_mutables:
            mixed_op = mutable.registered_module
            unused = {}
            if self.get_forward_mode() in ['full', 'two', 'full_v2']:
                involved_index = mixed_op.active_index + mixed_op.inactive_index
            else:
                involved_index = mixed_op.active_index
            for i in range(mixed_op.n_choices):
                if i not in involved_index:
                    unused[i] = mutable.choices[i]
                    mutable.choices[i] = None
            self._unused_modules.append(unused)

    def unused_modules_back(self):
        if self._unused_modules is None:
            return
        for m, unused in zip(self.mutable_list, self._unused_modules):
            for i in unused:
                m.choices[i] = unused[i]
        self._unused_modules = None

    def arch_requires_grad(self):
        for mutable in self.undedup_mutables:
            mutable.registered_module.to_requires_grad()

    def arch_disable_grad(self):
        for mutable in self.undedup_mutables:
            mutable.registered_module.disable_grad()

    def sample_final(self):
        result = dict()
        for mutable in self.undedup_mutables:
            assert isinstance(mutable, LayerChoice)
            index, _ = mutable.registered_module.chosen_index
            result[mutable.key] = F.one_hot(torch.tensor(index), num_classes=mutable.length).view(-1).bool()
        return result