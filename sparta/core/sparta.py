import torch
import torch.nn as nn

from sparta.propagation import propagate_sparsity
from sparta.transformation import optimize_and_rebuild

class SpartaModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.opt_modules = None
        self.opt_model = None
        # optimize the model based on its sparsity
        self.opt_model = self._optimize_sparsity()

    def _optimize_sparsity(self):
        # sparsity attributes are inplace updated in self.model
        post_sparsity = propagate_sparsity(self.model)
        # transformation, specialization, and rebuild model
        opt_model = optimize_and_rebuild(self.model,
                                         post_sparsity,
                                         backend='pytorch',
                                         device_info=None)
        return opt_model

    def forward(self, *inputs):
        return self.opt_model(*inputs)