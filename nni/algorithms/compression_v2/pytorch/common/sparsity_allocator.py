from typing import Dict, List, Union, Optional

import torch
from torch import Tensor

from nni.algorithms.compression_v2.pytorch.base.pruner import Pruner
from nni.algorithms.compression_v2.pytorch.base.common import SparsityAllocator


class NormalSparsityAllocator(SparsityAllocator):
    def __init__(self, pruner: Pruner, dim: Optional[Union[int, List[int]]] = None):
        """
        Parameters
        ----------
        pruner
            The pruner that wrapped the module.
        dim
            The dimensions that corresponding to the metric, None means one-to-one correspondence.
        """
        super().__init__(pruner)
        self.dim = dim if not isinstance(dim, int) else [dim]
        if self.dim is not None:
            assert all(i >= 0 for i in self.dim)
            self.dim = sorted(self.dim)

    def generate_sparsity(self, metrics: Dict[str, Tensor]) -> Dict[str, Dict[str, Tensor]]:
        masks = {}
        for name, wrapper in self.pruner._get_modules_wrapper().items():
            sparsity_rate = wrapper.config['sparsity']

            assert name in metrics, 'Metric of %s is not calculated.'
            metric = metrics[name]
            prune_num = int(sparsity_rate * metric.numel())
            if prune_num == 0:
                continue
            threshold = torch.topk(metric.view(-1), prune_num, largest=False)[0].max()
            mask = torch.gt(metric, threshold).type_as(metric)
            weight_size = wrapper.module.weight.data.size()
            if self.dim is None:
                assert len(mask.size()) == len(weight_size)
                masks[name] = {'weight_mask': mask}
            else:
                # expand mask to weight size
                assert len(mask.size()) == len(self.dim)
                assert all(weight_size[j] == mask.size()[i] for i, j in enumerate(self.dim))
                idxs = list(range(len(weight_size)))
                [idxs.pop(i) for i in reversed(self.dim)]
                weight_mask = mask.clone()
                for i in idxs:
                    weight_mask = weight_mask.unsqueeze(i)
                masks[name] = {'weight_mask': weight_mask.expand(weight_size).clone()}
                # NOTE: assume we only mask output
                if wrapper.bias_mask is not None and mask.size() == wrapper.bias_mask.size():
                    masks[name]['bias_mask'] = mask.clone()
        return masks
