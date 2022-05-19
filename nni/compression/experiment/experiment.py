# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations
import logging
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, List

import torch
from torch.nn import Module
from torch.optim import Optimizer

from nni.compression.experiment.config import generate_compression_search_space
from nni.experiment import Experiment
from .config import CompressionExperimentConfig, CompressionVessel

_logger = logging.getLogger('nni.experiment')


class CompressionExperiment(Experiment):
    """
    Note: This is an experimental feature, the interface is not stable.

    Parameters
    ----------
    config_or_platform
        A `CompressionExperimentConfig` or the training service name or list of the training service name or None.
    model
        The pytorch model wanted to compress.
    finetuner
        The finetuner handled all finetune logic, use a pytorch module as input.
    evaluator
        Evaluate the pruned model and give a score.
    dummy_input
        It is used by `torch.jit.trace` to trace the model.
    trainer
        A callable function used to train model or just inference. Take model, optimizer, criterion as input.
        Note that the model should only trained or inferenced one epoch in the trainer.

        Example::

            def trainer(model: Module, optimizer: Optimizer, criterion: Callable[[Tensor, Tensor], Tensor]):
                training = model.training
                model.train(mode=True)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    # If you don't want to update the model, you can skip `optimizer.step()`, and set train mode False.
                    optimizer.step()
                model.train(mode=training)
    optimizer
        The traced optimizer instance which the optimizer class is wrapped by nni.trace.
        E.g. ``traced_optimizer = nni.trace(torch.nn.Adam)(model.parameters())``.
    criterion
        The criterion function used in trainer. Take model output and target value as input, and return the loss.
    device
        The selected device.
    """

    # keep this interface for now, will change after support lightning
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

        self.temp_directory = tempfile.mkdtemp(prefix='nni_compression_{}_'.format(self.id))
        torch.save(model.state_dict(), Path(self.temp_directory, 'model_state_dict.pth'))
        self.vessel = CompressionVessel(model, finetuner, evaluator, dummy_input, trainer, optimizer, criterion, device)

    def start(self, port: int = 8080, debug: bool = False) -> None:
        # TODO: python3 is not robust, need support in nni manager
        self.config.trial_command = 'python3 -m nni.compression.experiment.trial_entry'

        # copy files in temp directory to nni_outputs/checkpoint
        # TODO: copy files to code dir is a temporary solution, need nnimanager support upload multi-directory,
        # or package additional files when uploading.
        checkpoint_dir = Path(self.config.trial_code_directory, 'nni_outputs', 'checkpoint')
        shutil.copytree(self.temp_directory, checkpoint_dir, dirs_exist_ok=True)

        if self.config.search_space or self.config.search_space_file:
            _logger.warning('Manual configuration of search_space is not recommended in compression experiments. %s',
                            'Please make sure you know what will happen.')
        else:
            self.config.search_space = generate_compression_search_space(self.config.compression_setting, self.vessel)

        return super().start(port, debug)
