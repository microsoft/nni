from typing import Dict, List, Optional, Union

from torch import Tensor

from nni.algorithms.compression_v2.pytorch.base.compressor import Compressor


class DataCollector:
    def __init__(self, compressor: Compressor):
        self.compressor = compressor

    def reset(self):
        raise NotImplementedError()

    def collect(self) -> Dict:
        raise NotImplementedError()


class MetricsCalculator:
    def calculate_metrics(self, data: Dict) -> Dict[str, Tensor]:
        raise NotImplementedError()


class SparsityAllocator:
    def __init__(self, pruner: Compressor, dim: Optional[Union[int, List[int]]] = None):
        """
        Parameters
        ----------
        pruner
            The pruner that wrapped the module.
        dim
            The dimensions that corresponding to the metric, None means one-to-one correspondence.
        """
        self.pruner = pruner
        self.dim = dim if not isinstance(dim, int) else [dim]
        if self.dim is not None:
            assert all(i >= 0 for i in self.dim)
            self.dim = sorted(self.dim)

    def generate_sparsity(self, metrics: Dict) -> Dict[str, Dict[str, Tensor]]:
        raise NotImplementedError()

    def _expand_mask_with_dim(self, name: str, mask: Tensor) -> Dict[str, Tensor]:
        wrapper = self.pruner._get_modules_wrapper()[name]
        weight_size = wrapper.module.weight.data.size()
        if self.dim is None:
            assert len(mask.size()) == len(weight_size)
            expand_mask = {'weight_mask': mask}
        else:
            # expand mask to weight size
            assert len(mask.size()) == len(self.dim)
            assert all(weight_size[j] == mask.size()[i] for i, j in enumerate(self.dim))
            idxs = list(range(len(weight_size)))
            [idxs.pop(i) for i in reversed(self.dim)]
            weight_mask = mask.clone()
            for i in idxs:
                weight_mask = weight_mask.unsqueeze(i)
            expand_mask = {'weight_mask': weight_mask.expand(weight_size).clone()}
            # NOTE: assume we only mask output
            if wrapper.bias_mask is not None and mask.size() == wrapper.bias_mask.size():
                expand_mask['bias_mask'] = mask.clone()
        return expand_mask
