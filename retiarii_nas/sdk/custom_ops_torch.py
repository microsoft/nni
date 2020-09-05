import torch
import torch.nn as nn

class View(nn.Module):
    def __init__(self, shape=(-1, )):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class Placeholder(nn.Module):
    def __init__(self, name, shape=None):
        super().__init__()
        self.name = name
        self.shape = shape

    def forward(self, x):
        if self.shape is None:
            val = x.size(0)
            return torch.full(x.size(), val).to(x.device)
        else:
            val = x.size(0)
            return torch.full(self.shape, val).to(x.device)
