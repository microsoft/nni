import logging
from typing import Any, Callable, List, Optional, Union
from torch.nn import Module
from torch.optim import Optimizer

from nni.compression.experiment.config.utils import cc_cv2ss
from nni.experiment import Experiment, ExperimentConfig
from .config import CompressionConfig, CompressionVessel

_logger = logging.getLogger('nni.experiment')


class CompressionExperiment(Experiment):
    def __init__(self, config_or_platform: Union[ExperimentConfig, str, List[str]],
                 compression_config: CompressionConfig, model: Module, finetuner: Callable[[Module], None],
                 evaluator: Callable[[Module], float], dummy_input: Optional[Any],
                 trainer: Optional[Callable[[Module, Optimizer, Callable[[Any, Any], Any]], None]],
                 optimizer: Optional[Optimizer], criterion: Optional[Callable[[Any, Any], Any]],
                 device: str):
        super().__init__(config_or_platform)

        self.compression_config = compression_config
        assert all([model, finetuner, evaluator])
        assert all([trainer, optimizer, criterion]) or not Any([trainer, optimizer, criterion])
        self.vessel = CompressionVessel(model, finetuner, evaluator, dummy_input, trainer, optimizer, criterion, device)

    def start(self, port: int = 8080, debug: bool = False) -> None:
        if self.config.search_space or self.config.search_space_file:
            _logger.warning('Manual configuration of search_space is not recommended in compression experiments. ' + \
                            'Please make sure you know what will happen.')
        else:
            self.config.search_space = cc_cv2ss(self.compression_config, self.vessel)
        self.config.trial_command = 'python3 -m nni.compression.experiment.trial_entry'
        return super().start(port, debug)
