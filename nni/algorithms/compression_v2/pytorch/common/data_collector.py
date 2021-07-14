import logging
from typing import Dict

from torch import Tensor

from nni.algorithms.compression_v2.pytorch.base.common import DataCollector, TrainerBasedDataCollector

_logger = logging.getLogger(__name__)


class WeightDataCollector(DataCollector):
    def reset(self):
        pass

    def collect(self) -> Dict[str, Tensor]:
        data = {}
        for _, wrapper in self.compressor._get_modules_wrapper().items():
            data[wrapper.name] = wrapper.module.weight.data.clone().detach()
        return data


class WeightTrainerBasedDataCollector(TrainerBasedDataCollector):
    def collect(self) -> Dict:
        for _ in range(self.training_epochs):
            self.trainer(self.compressor.bound_model, self.optimizer, self.criterion)

        data = {}
        for _, wrapper in self.compressor._get_modules_wrapper().items():
            data[wrapper.name] = wrapper.module.weight.data.clone().detach()
        return data


class SingleHookTrainerBasedDataCollector(TrainerBasedDataCollector):
    def collect(self) -> Dict:
        for _ in range(self.training_epochs):
            self.trainer(self.compressor.bound_model, self.optimizer, self.criterion)

        data = {}
        [data.update(buffer_dict) for _, buffer_dict in self._hook_buffer.items()]
        return data
