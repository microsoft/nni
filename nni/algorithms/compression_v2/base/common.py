from typing import Dict

from torch import Tensor

from compression_v2.base.compressor import Compressor


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
    def __init__(self, pruner: Compressor):
        self.pruner = pruner

    def generate_sparsity(self, metrics: Dict) -> Dict[str, Dict[str, Tensor]]:
        raise NotImplementedError()
