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

    def __init__(self, **kwargs):
        super().__init__()
        for item_key, item_value in kwargs.items():
            if item_key == "key":
                self.key = kwargs.get(item_key, self.__class__.__name__ + str(global_mutable_counting()))
            else:
                self.__setattr__(item_key, item_value)

        self.name = self.key

    def __deepcopy__(self, memodict=None):
        raise NotImplementedError

    def set_mutator(self, mutator):
        self.__dict__["mutator"] = mutator

    def forward(self, *inputs):
        raise NotImplementedError("Mutable forward must be implemented")

    # def __repr__(self):
    #     return "{} ({})".format(self.name, self.key)

    def similar(self, other):
        return self == other


class LayerChoice(Mutable):
    def __init__(self, ops, **kwargs):
        super().__init__(**kwargs)
        self.length = len(ops)
        self.choices = nn.ModuleList(ops)

    def forward(self, *inputs):
        return self.mutator.on_forward(self, self.choices, *inputs)

    def similar(self, other):
        return type(self) == type(other) and self.length == other.length


class InputChoice(Mutable):
    def __init__(self, n_candidates, n_selected=None, reduction="mean", return_index=False, **kwargs):
        super().__init__(**kwargs)
        self.n_candidates = n_candidates
        self.n_selected = n_selected
        self.reduction = reduction
        self.return_index = return_index

    def forward(self, *inputs):
        assert len(
            inputs) == self.n_candidates, "Length of the input list must be equal to number of candidates."
        return self.mutator.on_forward(self, *inputs)

    def similar(self, other):
        return type(self) == type(other) and \
            self.n_candidates == other.n_candidates and self.n_selected and other.n_selected
