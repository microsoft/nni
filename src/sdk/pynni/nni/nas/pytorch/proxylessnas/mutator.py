# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from nni.nas.pytorch.base_mutator import BaseMutator
from nni.nas.pytorch.mutables import LayerChoice
from .utils import detach_variable

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
    This class is to instantiate and manage info of one LayerChoice.
    It includes architecture weights, binary weights, and member functions
    operating the weights.

    forward_mode:
        forward/backward mode for LayerChoice: None, two, full, and full_v2.
        For training architecture weights, we use full_v2 by default, and for training
        model weights, we use None.
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
        self.ap_path_alpha = nn.Parameter(torch.Tensor(len(mutable)))
        self.ap_path_wb = nn.Parameter(torch.Tensor(len(mutable)))
        self.ap_path_alpha.requires_grad = False
        self.ap_path_wb.requires_grad = False
        self.active_index = [0]
        self.inactive_index = None
        self.log_prob = None
        self.current_prob_over_ops = None
        self.n_choices = len(mutable)

    def get_ap_path_alpha(self):
        return self.ap_path_alpha

    def to_requires_grad(self):
        self.ap_path_alpha.requires_grad = True
        self.ap_path_wb.requires_grad = True

    def to_disable_grad(self):
        self.ap_path_alpha.requires_grad = False
        self.ap_path_wb.requires_grad = False

    def forward(self, mutable, x):
        """
        Define forward of LayerChoice. For 'full_v2', backward is also defined.
        The 'two' mode is explained in section 3.2.1 in the paper.
        The 'full_v2' mode is explained in Appendix D in the paper.

        Parameters
        ----------
        mutable : LayerChoice
            this layer's mutable
        x : tensor
            inputs of this layer, only support one input

        Returns
        -------
        output: tensor
            output of this layer
        """
        if MixedOp.forward_mode == 'full' or MixedOp.forward_mode == 'two':
            output = 0
            for _i in self.active_index:
                oi = self.candidate_ops[_i](x)
                output = output + self.ap_path_wb[_i] * oi
            for _i in self.inactive_index:
                oi = self.candidate_ops[_i](x)
                output = output + self.ap_path_wb[_i] * oi.detach()
        elif MixedOp.forward_mode == 'full_v2':
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
                x, self.ap_path_wb, run_function(mutable.key, list(mutable), self.active_index[0]),
                backward_function(mutable.key, list(mutable), self.active_index[0], self.ap_path_wb))
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
        probs = F.softmax(self.ap_path_alpha, dim=0)  # softmax to probability
        return probs

    @property
    def chosen_index(self):
        """
        choose the op with max prob

        Returns
        -------
        int
            index of the chosen one
        numpy.float32
            prob of the chosen one
        """
        probs = self.probs_over_ops.data.cpu().numpy()
        index = int(np.argmax(probs))
        return index, probs[index]

    def active_op(self, mutable):
        """
        assume only one path is active

        Returns
        -------
        PyTorch module
            the chosen operation
        """
        return mutable[self.active_index[0]]

    @property
    def active_op_index(self):
        """
        return active op's index, the active op is sampled

        Returns
        -------
        int
            index of the active op
        """
        return self.active_index[0]

    def set_chosen_op_active(self):
        """
        set chosen index, active and inactive indexes
        """
        chosen_idx, _ = self.chosen_index
        self.active_index = [chosen_idx]
        self.inactive_index = [_i for _i in range(0, chosen_idx)] + \
                              [_i for _i in range(chosen_idx + 1, self.n_choices)]

    def binarize(self, mutable):
        """
        Sample based on alpha, and set binary weights accordingly.
        ap_path_wb is set in this function, which is called binarize.

        Parameters
        ----------
        mutable : LayerChoice
            this layer's mutable
        """
        self.log_prob = None
        # reset binary gates
        self.ap_path_wb.data.zero_()
        probs = self.probs_over_ops
        if MixedOp.forward_mode == 'two':
            # sample two ops according to probs
            sample_op = torch.multinomial(probs.data, 2, replacement=False)
            probs_slice = F.softmax(torch.stack([
                self.ap_path_alpha[idx] for idx in sample_op
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
            self.ap_path_wb.data[active_op] = 1.0
        else:
            sample = torch.multinomial(probs, 1)[0].item()
            self.active_index = [sample]
            self.inactive_index = [_i for _i in range(0, sample)] + \
                                [_i for _i in range(sample + 1, len(mutable))]
            self.log_prob = torch.log(probs[sample])
            self.current_prob_over_ops = probs
            self.ap_path_wb.data[sample] = 1.0
        # avoid over-regularization
        for choice in mutable:
            for _, param in choice.named_parameters():
                param.grad = None

    @staticmethod
    def delta_ij(i, j):
        if i == j:
            return 1
        else:
            return 0

    def set_arch_param_grad(self, mutable):
        """
        Calculate alpha gradient for this LayerChoice.
        It is calculated using gradient of binary gate, probs of ops.
        """
        binary_grads = self.ap_path_wb.grad.data
        if self.active_op(mutable).is_zero_layer():
            self.ap_path_alpha.grad = None
            return
        if self.ap_path_alpha.grad is None:
            self.ap_path_alpha.grad = torch.zeros_like(self.ap_path_alpha.data)
        if MixedOp.forward_mode == 'two':
            involved_idx = self.active_index + self.inactive_index
            probs_slice = F.softmax(torch.stack([
                self.ap_path_alpha[idx] for idx in involved_idx
            ]), dim=0).data
            for i in range(2):
                for j in range(2):
                    origin_i = involved_idx[i]
                    origin_j = involved_idx[j]
                    self.ap_path_alpha.grad.data[origin_i] += \
                        binary_grads[origin_j] * probs_slice[j] * (MixedOp.delta_ij(i, j) - probs_slice[i])
            for _i, idx in enumerate(self.active_index):
                self.active_index[_i] = (idx, self.ap_path_alpha.data[idx].item())
            for _i, idx in enumerate(self.inactive_index):
                self.inactive_index[_i] = (idx, self.ap_path_alpha.data[idx].item())
        else:
            probs = self.probs_over_ops.data
            for i in range(self.n_choices):
                for j in range(self.n_choices):
                    self.ap_path_alpha.grad.data[i] += binary_grads[j] * probs[j] * (MixedOp.delta_ij(i, j) - probs[i])
        return

    def rescale_updated_arch_param(self):
        """
        rescale architecture weights for the 'two' mode.
        """
        if not isinstance(self.active_index[0], tuple):
            assert self.active_op.is_zero_layer()
            return
        involved_idx = [idx for idx, _ in (self.active_index + self.inactive_index)]
        old_alphas = [alpha for _, alpha in (self.active_index + self.inactive_index)]
        new_alphas = [self.ap_path_alpha.data[idx] for idx in involved_idx]

        offset = math.log(
            sum([math.exp(alpha) for alpha in new_alphas]) / sum([math.exp(alpha) for alpha in old_alphas])
        )

        for idx in involved_idx:
            self.ap_path_alpha.data[idx] -= offset


class ProxylessNasMutator(BaseMutator):
    """
    This mutator initializes and operates all the LayerChoices of the input model.
    It is for the corresponding trainer to control the training process of LayerChoices,
    coordinating with whole training process.
    """
    def __init__(self, model):
        """
        Init a MixedOp instance for each mutable i.e., LayerChoice.
        And register the instantiated MixedOp in corresponding LayerChoice.
        If does not register it in LayerChoice, DataParallel does not work then,
        because architecture weights are not included in the DataParallel model.
        When MixedOPs are registered, we use ```requires_grad``` to control
        whether calculate gradients of architecture weights.

        Parameters
        ----------
        model : pytorch model
            The model that users want to tune, it includes search space defined with nni nas apis
        """
        super(ProxylessNasMutator, self).__init__(model)
        self._unused_modules = None
        self.mutable_list = []
        for mutable in self.undedup_mutables:
            self.mutable_list.append(mutable)
            mutable.registered_module = MixedOp(mutable)

    def on_forward_layer_choice(self, mutable, *args, **kwargs):
        """
        Callback of layer choice forward. This function defines the forward
        logic of the input mutable. So mutable is only interface, its real
        implementation is defined in mutator.

        Parameters
        ----------
        mutable: LayerChoice
            forward logic of this input mutable
        args: list of torch.Tensor
            inputs of this mutable
        kwargs: dict
            inputs of this mutable

        Returns
        -------
        torch.Tensor
            output of this mutable, i.e., LayerChoice
        int
            index of the chosen op
        """
        # FIXME: return mask, to be consistent with other algorithms
        idx = mutable.registered_module.active_op_index
        return mutable.registered_module(mutable, *args, **kwargs), idx

    def reset_binary_gates(self):
        """
        For each LayerChoice, binarize binary weights
        based on alpha to only activate one op.
        It traverses all the mutables in the model to do this.
        """
        for mutable in self.undedup_mutables:
            mutable.registered_module.binarize(mutable)

    def set_chosen_op_active(self):
        """
        For each LayerChoice, set the op with highest alpha as the chosen op.
        Usually used for validation.
        """
        for mutable in self.undedup_mutables:
            mutable.registered_module.set_chosen_op_active()

    def num_arch_params(self):
        """
        The number of mutables, i.e., LayerChoice

        Returns
        -------
        int
            the number of LayerChoice in user model
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
        Get all the architecture parameters.

        yield
        -----
        PyTorch Parameter
            Return ap_path_alpha of the traversed mutable
        """
        for mutable in self.undedup_mutables:
            yield mutable.registered_module.get_ap_path_alpha()

    def change_forward_mode(self, mode):
        """
        Update forward mode of MixedOps, as training architecture weights and
        model weights use different forward modes.
        """
        MixedOp.forward_mode = mode

    def get_forward_mode(self):
        """
        Get forward mode of MixedOp

        Returns
        -------
        string
            the current forward mode of MixedOp
        """
        return MixedOp.forward_mode

    def rescale_updated_arch_param(self):
        """
        Rescale architecture weights in 'two' mode.
        """
        for mutable in self.undedup_mutables:
            mutable.registered_module.rescale_updated_arch_param()

    def unused_modules_off(self):
        """
        Remove unused modules for each mutables.
        The removed modules are kept in ```self._unused_modules``` for resume later.
        """
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
                    unused[i] = mutable[i]
                    mutable[i] = None
            self._unused_modules.append(unused)

    def unused_modules_back(self):
        """
        Resume the removed modules back.
        """
        if self._unused_modules is None:
            return
        for m, unused in zip(self.mutable_list, self._unused_modules):
            for i in unused:
                m[i] = unused[i]
        self._unused_modules = None

    def arch_requires_grad(self):
        """
        Make architecture weights require gradient
        """
        for mutable in self.undedup_mutables:
            mutable.registered_module.to_requires_grad()

    def arch_disable_grad(self):
        """
        Disable gradient of architecture weights, i.e., does not
        calcuate gradient for them.
        """
        for mutable in self.undedup_mutables:
            mutable.registered_module.to_disable_grad()

    def sample_final(self):
        """
        Generate the final chosen architecture.

        Returns
        -------
        dict
            the choice of each mutable, i.e., LayerChoice
        """
        result = dict()
        for mutable in self.undedup_mutables:
            assert isinstance(mutable, LayerChoice)
            index, _ = mutable.registered_module.chosen_index
            # pylint: disable=not-callable
            result[mutable.key] = F.one_hot(torch.tensor(index), num_classes=len(mutable)).view(-1).bool()
        return result
