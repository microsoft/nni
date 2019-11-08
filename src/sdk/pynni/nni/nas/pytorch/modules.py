
from torch import nn as nn


class RankedModule(nn.Module):
    def  __init__(self, rank=None):
        super(RankedModule, self).__init__()
        self.rank = rank
