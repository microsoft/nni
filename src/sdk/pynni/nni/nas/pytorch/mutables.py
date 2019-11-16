import torch.nn as nn

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


class LayerChoice(Mutable):
    def __init__(self, op_candidates, reduction="mean", return_mask=False, key=None):
        super().__init__(key=key)
        self.length = len(op_candidates)
        self.choices = nn.ModuleList(op_candidates)
        self.reduction = reduction
        self.return_mask = return_mask
    
    def __len__(self):
        return len(self.choices)

    def forward(self, *inputs):
        out, mask = self.mutator.on_forward_layer_choice(self, *inputs)
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
