import torch
from torch import nn as nn
from torch.nn import functional as F

from nni.nas.pytorch.mutables import LayerChoice
from nni.nas.pytorch.mutator import PyTorchMutator


class DartsMutator(PyTorchMutator):

    def before_build(self, model):
        self.choices = nn.ParameterDict()
        self.register_on_init_hook(LayerChoice, self.on_init_layer_choice)
        self.register_on_forward_hook(LayerChoice, self.on_forward_layer_choice)

    def on_init_layer_choice(self, mutable: LayerChoice):
        self.choices[mutable.key] = nn.Parameter(
            1.0E-3 * torch.randn(mutable.length))

    def on_calc_layer_choice_mask(self, mutable: LayerChoice):
        return F.softmax(self.choices[mutable.key], dim=-1)
