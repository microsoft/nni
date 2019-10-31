import torch
from torch import nn as nn
from torch.nn import functional as F

from nni.nas.pytorch.mutables import LayerChoice
from nni.nas.pytorch.mutator import PyTorchMutator


class DartsMutator(PyTorchMutator):

    def before_build(self, model):
        self.choices = nn.ParameterDict()

    def on_init_layer_choice(self, mutable: LayerChoice):
        self.choices[mutable.key] = nn.Parameter(1.0E-3 * torch.randn(mutable.length))

    def on_forward_layer_choice(self, mutable: LayerChoice, ops, *inputs):
        weights = F.softmax(self.choices[mutable.key], dim=-1)
        return sum(w * op(*inputs) for w, op in zip(weights, ops))
