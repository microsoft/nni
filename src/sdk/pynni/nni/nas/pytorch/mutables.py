import torch.nn as nn

from nni.nas.utils import global_mutable_counting


class PyTorchMutable(nn.Module):
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
            self.key = key
        else:
            self.key = self.__class__.__name__ + str(global_mutable_counting())
        self.name = self.key

    def __deepcopy__(self, memodict=None):
        raise NotImplementedError

    def set_mutator(self, mutator):
        self.__dict__["mutator"] = mutator

    def forward(self, *inputs):
        raise NotImplementedError("Mutable forward must be implemented")

    def __repr__(self):
        return "{} ({})".format(self.name, self.key)

    def similar(self, other):
        return self == other


class LayerChoice(PyTorchMutable):
    def __init__(self, ops, key=None):
        super().__init__(key=key)
        self.length = len(ops)
        self.choices = nn.ModuleList(ops)

    def forward(self, *inputs):
        return self.mutator.on_forward(self, self.choices, *inputs)

    def similar(self, other):
        return type(self) == type(other) and self.length == other.length


class InputChoice(PyTorchMutable):
    def __init__(self, n_candidates, n_selected=None, reduction="mean", return_index=False, key=None):
        super().__init__(key=key)
        self.n_candidates = n_candidates
        self.n_selected = n_selected
        self.reduction = reduction
        self.return_index = return_index

    def forward(self, *inputs):
        assert len(inputs) == self.n_candidates, "Length of the input list must be equal to number of candidates."
        return self.mutator.on_forward(self, *inputs)

    def similar(self, other):
        return type(self) == type(other) and \
               self.n_candidates == other.n_candidates and self.n_selected and other.n_selected
