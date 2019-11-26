# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

import torch.nn as nn

from nni.nas.pytorch.utils import global_mutable_counting

logger = logging.getLogger(__name__)


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
                logger.warning("Warning: key \"%s\" is not string, converted to string.", key)
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
        if "mutator" in self.__dict__:
            raise RuntimeError("`set_mutator` is called more than once. Did you parse the search space multiple times? "
                               "Or did you apply multiple fixed architectures?")
        self.__dict__["mutator"] = mutator

    def forward(self, *inputs):
        raise NotImplementedError

    @property
    def key(self):
        return self._key

    @property
    def name(self):
        return self._name if hasattr(self, "_name") else "_key"

    @name.setter
    def name(self, name):
        self._name = name

    def _check_built(self):
        if not hasattr(self, "mutator"):
            raise ValueError(
                "Mutator not set for {}. You might have forgotten to initialize and apply your mutator. "
                "Or did you initialize a mutable on the fly in forward pass? Move to `__init__` "
                "so that trainer can locate all your mutables. See NNI docs for more details.".format(self))

    def __repr__(self):
        return "{} ({})".format(self.name, self.key)


class MutableScope(Mutable):
    """
    Mutable scope marks a subgraph/submodule to help mutators make better decisions.
    Mutators get notified when a mutable scope is entered and exited. Mutators can override ``enter_mutable_scope``
    and ``exit_mutable_scope`` to catch corresponding events, and do status dump or update.
    MutableScope are also mutables that are listed in the mutables (search space).
    """

    def __init__(self, key):
        super().__init__(key=key)

    def __call__(self, *args, **kwargs):
        try:
            self._check_built()
            self.mutator.enter_mutable_scope(self)
            return super().__call__(*args, **kwargs)
        finally:
            self.mutator.exit_mutable_scope(self)


class LayerChoice(Mutable):
    def __init__(self, op_candidates, reduction="sum", return_mask=False, key=None):
        super().__init__(key=key)
        self.length = len(op_candidates)
        self.choices = nn.ModuleList(op_candidates)
        self.reduction = reduction
        self.return_mask = return_mask

    def forward(self, *inputs):
        out, mask = self.mutator.on_forward_layer_choice(self, *inputs)
        if self.return_mask:
            return out, mask
        return out


class InputChoice(Mutable):
    """
    Input choice selects `n_chosen` inputs from `choose_from` (contains `n_candidates` keys). For beginners,
    use `n_candidates` instead of `choose_from` is a safe option. To get the most power out of it, you might want to
    know about `choose_from`.

    The keys in `choose_from` can be keys that appear in past mutables, or ``NO_KEY`` if there are no suitable ones.
    The keys are designed to be the keys of the sources. To help mutators make better decisions,
    mutators might be interested in how the tensors to choose from come into place. For example, the tensor is the
    output of some operator, some node, some cell, or some module. If this operator happens to be a mutable (e.g.,
    ``LayerChoice`` or ``InputChoice``), it has a key naturally that can be used as a source key. If it's a
    module/submodule, it needs to be annotated with a key: that's where a ``MutableScope`` is needed.
    """

    NO_KEY = ""

    def __init__(self, n_candidates=None, choose_from=None, n_chosen=None,
                 reduction="sum", return_mask=False, key=None):
        """
        Initialization.

        Parameters
        ----------
        n_candidates : int
            Number of inputs to choose from.
        choose_from : list of str
            List of source keys to choose from. At least of one of `choose_from` and `n_candidates` must be fulfilled.
            If `n_candidates` has a value but `choose_from` is None, it will be automatically treated as `n_candidates`
            number of empty string.
        n_chosen : int
            Recommended inputs to choose. If None, mutator is instructed to select any.
        reduction : str
            `mean`, `concat`, `sum` or `none`.
        return_mask : bool
            If `return_mask`, return output tensor and a mask. Otherwise return tensor only.
        key : str
            Key of the input choice.
        """
        super().__init__(key=key)
        # precondition check
        assert n_candidates is not None or choose_from is not None, "At least one of `n_candidates` and `choose_from`" \
                                                                    "must be not None."
        if choose_from is not None and n_candidates is None:
            n_candidates = len(choose_from)
        elif choose_from is None and n_candidates is not None:
            choose_from = [self.NO_KEY] * n_candidates
        assert n_candidates == len(choose_from), "Number of candidates must be equal to the length of `choose_from`."
        assert n_candidates > 0, "Number of candidates must be greater than 0."
        assert n_chosen is None or 0 <= n_chosen <= n_candidates, "Expected selected number must be None or no more " \
                                                                  "than number of candidates."

        self.n_candidates = n_candidates
        self.choose_from = choose_from
        self.n_chosen = n_chosen
        self.reduction = reduction
        self.return_mask = return_mask

    def forward(self, optional_inputs):
        """
        Forward method of LayerChoice.

        Parameters
        ----------
        optional_inputs : list or dict
            Recommended to be a dict. As a dict, inputs will be converted to a list that follows the order of
            `choose_from` in initialization. As a list, inputs must follow the semantic order that is the same as
            `choose_from`.

        Returns
        -------
        tuple of torch.Tensor and torch.Tensor or torch.Tensor
        """
        optional_input_list = optional_inputs
        if isinstance(optional_inputs, dict):
            optional_input_list = [optional_inputs[tag] for tag in self.choose_from]
        assert isinstance(optional_input_list, list), \
            "Optional input list must be a list, not a {}.".format(type(optional_input_list))
        assert len(optional_inputs) == self.n_candidates, \
            "Length of the input list must be equal to number of candidates."
        out, mask = self.mutator.on_forward_input_choice(self, optional_input_list)
        if self.return_mask:
            return out, mask
        return out
