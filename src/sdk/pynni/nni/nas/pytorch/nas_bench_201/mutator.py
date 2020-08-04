import torch
from nni.nas.pytorch.mutator import Mutator
from nni.nas.pytorch.mutables import LayerChoice


class NASBench201Mutator(Mutator):
    def reset(self, matrix):
        result = dict()
        for mutable in self.mutables:
            if isinstance(mutable, LayerChoice):
                keywords_split = mutable.key.split("_")
                onehot = matrix[int(keywords_split[-2]), int(keywords_split[-1])]
                result[mutable.key] = torch.tensor(onehot, dtype=torch.bool)
        self._cache = result
