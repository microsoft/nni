import torch
import torch.nn.functional as F

from nni.nas.pytorch.mutator import Mutator
from nni.nas.pytorch.mutables import LayerChoice, InputChoice


class RandomMutator(Mutator):
    """
    Random mutator that samples a random candidate in the search space each time ``reset()``.
    It uses random function in PyTorch, so users can set seed in PyTorch to ensure deterministic behavior.
    """

    def sample_search(self):
        """
        Sample a random candidate.
        """
        result = dict()
        for mutable in self.mutables:
            if isinstance(mutable, LayerChoice):
                gen_index = torch.randint(high=len(mutable), size=(1, ))
                result[mutable.key] = F.one_hot(gen_index, num_classes=len(mutable)).view(-1).bool()
            elif isinstance(mutable, InputChoice):
                if mutable.n_chosen is None:
                    result[mutable.key] = torch.randint(high=2, size=(mutable.n_candidates,)).view(-1).bool()
                else:
                    perm = torch.randperm(mutable.n_candidates)
                    mask = [i in perm[:mutable.n_chosen] for i in range(mutable.n_candidates)]
                    result[mutable.key] = torch.tensor(mask, dtype=torch.bool)  # pylint: disable=not-callable
        return result

    def sample_final(self):
        """
        Same as :meth:`sample_search`.
        """
        return self.sample_search()
