# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Callable, List

import torch
from torch.nn import Module
from torch.optim import Optimizer

from nni.compression.experiment.config import generate_compression_search_space
from nni.experiment import Experiment
from .config import CompressionExperimentConfig, CompressionVessel

_logger = logging.getLogger('nni.experiment')


class CompressionExperiment(Experiment):
    def __init__(self, config_or_platform: CompressionExperimentConfig | str | List[str] | None,
                 model: Module, finetuner: Callable[[Module], None],
                 evaluator: Callable[[Module], float], dummy_input: Any | None,
                 trainer: Callable[[Module, Optimizer, Callable[[Any, Any], Any]], None] | None,
                 optimizer: Optimizer | None, criterion: Callable[[Any, Any], Any] | None,
                 device: str | torch.device):
        super().__init__(config_or_platform)

        # have some risks if Experiment change its __init__, but work well for current version
        self.config: CompressionExperimentConfig | None = None
        if isinstance(config_or_platform, (str, list)):
            self.config = CompressionExperimentConfig(config_or_platform)
        else:
            self.config = config_or_platform

        assert all([model, finetuner, evaluator])
        assert all([trainer, optimizer, criterion]) or not any([trainer, optimizer, criterion])
        torch.save(model.state_dict(), Path(self.config.trial_code_directory, 'nni_model_state_dict.pth'))
        self.vessel = CompressionVessel(model, finetuner, evaluator, dummy_input, trainer, optimizer, criterion, device)

    def start(self, port: int = 8080, debug: bool = False) -> None:
        if self.config.search_space or self.config.search_space_file:
            _logger.warning('Manual configuration of search_space is not recommended in compression experiments. %s',
                            'Please make sure you know what will happen.')
        else:
            self.config.search_space = generate_compression_search_space(self.config.compression_setting, self.vessel)
        # TODO: python3 is not robust, need support in nni manager
        self.config.trial_command = 'python3 -m nni.compression.experiment.trial_entry'
        return super().start(port, debug)
