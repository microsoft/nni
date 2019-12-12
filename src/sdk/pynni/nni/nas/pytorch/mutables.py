import torch
import torch.nn as nn
from torch.nn import functional as F

from nni.nas.utils import global_mutable_counting


class Mutable(nn.Module):
    """
    Mutable is designed to function as a normal layer, with all necessary operators' weights.
    States and weights of architectures should be included in mutator, instead of the layer itself.

    Mutable has a key, which marks the identity of the mutable. This key can be used by users to share
    decisions among different mutables. In mutator's implementation, mutators should use the key to
    distinguish different mutables. Mutables that share the same key should be "similar" to each other.

    Currently the default scope for keys is global.
    """

    def __init__(self, key=None):
        super().__init__()
        if key is not None:
            if not isinstance(key, str):
                key = str(key)
                print("Warning: key \"{}\" is not string, converted to string.".format(key))
            self._key = key
        else:
            self._key = self.__class__.__name__ + str(global_mutable_counting())
        self.init_hook = self.forward_hook = None

    def __deepcopy__(self, memodict=None):
        raise NotImplementedError("Deep copy doesn't work for mutables.")

    def __call__(self, *args, **kwargs):
        self._check_built()
        return super().__call__(*args, **kwargs)

    def set_mutator(self, mutator):
        self.__dict__["mutator"] = mutator

    def forward(self, *inputs):
        raise NotImplementedError("Mutable forward must be implemented.")

    @property
    def key(self):
        return self._key

    @property
    def name(self):
        return self._name if hasattr(self, "_name") else "_key"

    @name.setter
    def name(self, name):
        self._name = name

    def similar(self, other):
        return type(self) == type(other)

    def _check_built(self):
        if not hasattr(self, "mutator"):
            raise ValueError(
                "Mutator not set for {}. Did you initialize a mutable on the fly in forward pass? Move to __init__"
                "so that trainer can locate all your mutables. See NNI docs for more details.".format(self))

    def __repr__(self):
        return "{} ({})".format(self.name, self.key)


class MutableScope(Mutable):
    """
    Mutable scope labels a subgraph to help mutators make better decisions. Mutators get notified when a mutable scope
    is entered and exited. Mutators can override ``enter_mutable_scope`` and ``exit_mutable_scope`` to catch
    corresponding events, and do status dump or update.
    """

    def __init__(self, key):
        super().__init__(key=key)

    def build(self):
        self.mutator.on_init_mutable_scope(self)

    def __call__(self, *args, **kwargs):
        try:
            self.mutator.enter_mutable_scope(self)
            return super().__call__(*args, **kwargs)
        finally:
            self.mutator.exit_mutable_scope(self)


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
        #self.mutable = mutable
        self.AP_path_alpha = nn.Parameter(torch.Tensor(mutable.length))
        self.AP_path_wb = nn.Parameter(torch.Tensor(mutable.length))
        self.active_index = [0]
        self.inactive_index = None
        self.log_prob = None
        self.current_prob_over_ops = None
        self.n_choices = mutable.length

    def get_AP_path_alpha(self):
        return self.AP_path_alpha

    def forward(self, key, choices, x):
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
            x, self.AP_path_wb, run_function(key, choices, self.active_index[0]),
            backward_function(key, choices, self.active_index[0], self.AP_path_wb))
        return output

    '''def forward(self, mutable, x):
        out = mutable.choices[0](x)
        for choice in mutable.choices[1:]:
            out += choice(x)
        return out'''

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

    '''def active_op(self, mutable):
        """ assume only one path is active """
        return mutable.choices[self.active_index[0]]'''

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

    def set_arch_param_grad(self, mutable_len):
        """
        Calculate alpha gradient for this LayerChoice
        """
        binary_grads = self.AP_path_wb.grad.data
        '''if self.active_op(mutable).is_zero_layer():
            self.AP_path_alpha.grad = None
            return'''
        if self.AP_path_alpha.grad is None:
            self.AP_path_alpha.grad = torch.zeros_like(self.AP_path_alpha.data)
        probs = self.probs_over_ops.data
        for i in range(mutable_len):
            for j in range(mutable_len):
                self.AP_path_alpha.grad.data[i] += binary_grads[j] * probs[j] * (self._delta_ij(i, j) - probs[i])


class LayerChoice(Mutable):
    def __init__(self, op_candidates, reduction="mean", return_mask=False, key=None):
        super().__init__(key=key)
        self.length = len(op_candidates)
        self.choices = nn.ModuleList(op_candidates)
        self.reduction = reduction
        self.return_mask = return_mask
        self.registered_module = MixedOp(self)

    def __len__(self):
        return len(self.choices)

    #def register_module(self, module):
    #    self.registered_module = module

    def forward(self, *inputs):
        out = self.registered_module(self.key, self.choices, *inputs)
        #mask = self.mutator.on_forward_layer_choice(self, *inputs)
        mask = None
        if self.return_mask:
            return out, mask
        return out

    def similar(self, other):
        return type(self) == type(other) and self.length == other.length


class InputChoice(Mutable):
    def __init__(self, n_candidates, n_selected=None, reduction="mean", return_mask=False, key=None):
        super().__init__(key=key)
        assert n_candidates > 0, "Number of candidates must be greater than 0."
        self.n_candidates = n_candidates
        self.n_selected = n_selected
        self.reduction = reduction
        self.return_mask = return_mask

    def build(self):
        self.mutator.on_init_input_choice(self)

    def forward(self, optional_inputs, tags=None):
        assert len(optional_inputs) == self.n_candidates, \
            "Length of the input list must be equal to number of candidates."
        if tags is None:
            tags = [""] * self.n_candidates
        else:
            assert len(tags) == self.n_candidates, "Length of tags must be equal to number of candidates."
        out, mask = self.mutator.on_forward_input_choice(self, optional_inputs, tags)
        if self.return_mask:
            return out, mask
        return out

    def similar(self, other):
        return type(self) == type(other) and \
               self.n_candidates == other.n_candidates and self.n_selected and other.n_selected
