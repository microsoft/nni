import torch
from torch import nn as nn
from torch.nn import functional as F

from nni.nas.pytorch.mutables import LayerChoice
from nni.nas.pytorch.darts import DartsMutator
from nni.nas.pytorch.darts import DartsTrainer


class PdartsMutator(DartsMutator):

    def on_forward_layer_choice(self, mutable: LayerChoice, ops, *inputs):
        output = super().on_forward_layer_choice(self, mutable, ops, *inputs)
        self.output_shape = output.shape()
        print(self.output_shape)
        return output
