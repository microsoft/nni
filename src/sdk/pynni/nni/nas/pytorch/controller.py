import torch.nn as nn


class Controller(nn.Module):
    def build(self, mutables):
        raise NotImplementedError

    def sample_search(self, mutables):
        raise NotImplementedError

    def sample_final(self, mutables):
        raise NotImplementedError
