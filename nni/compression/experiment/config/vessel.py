# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import base64
import io
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union, overload

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from nni.algorithms.compression.v2.pytorch.utils.constructor_helper import OptimizerConstructHelper
from nni.common import dump, load
from nni.experiment.config.base import ConfigBase


@dataclass(init=False)
class CompressionVessel(ConfigBase):
    """
    This is an internal class that helps serialize model-related parameters during model compression.

    # FIXME: In fact, it is not a `Config`, the only reason it is a `Config` right now is that its data attribute
    # will go into the search space as a single choice field. Need to refactor after the experiment config is stable.
    """

    model: str
    finetuner: str
    evaluator: str
    dummy_input: str
    trainer: Optional[str]
    optimizer_helper: Optional[str]
    criterion: Optional[str]
    device: str

    @overload
    def __init__(self, model: str, finetuner: str, evaluator: str, dummy_input: str,
                 trainer: str, optimizer_helper: str, criterion: str, device: str):
        ...

    @overload
    def __init__(self, model: Module, finetuner: Callable[[Module], None], evaluator: Callable[[Module], float],
                 dummy_input: Tensor, trainer: Optional[Callable[[Module, Optimizer, Callable[[Any, Any], Any]], None]],
                 optimizer_helper: Union[Optimizer, OptimizerConstructHelper, None],
                 criterion: Optional[Callable[[Any, Any], Any]], device: Union[str, torch.device]):
        ...

    def __init__(self,
                 model: Union[Module, str],
                 finetuner: Union[Callable[[Module], None], str],
                 evaluator: Union[Callable[[Module], float], str],
                 dummy_input: Union[Tensor, str],
                 trainer: Union[Callable[[Module, Optimizer, Callable[[Any, Any], Any]], None], str, None],
                 optimizer_helper: Union[Optimizer, OptimizerConstructHelper, str, None],
                 criterion: Union[Callable[[Any, Any], Any], str, None],
                 device: Union[torch.device, str]):
        self.model = dump(model) if not isinstance(model, str) else model
        self.finetuner = dump(finetuner) if not isinstance(finetuner, str) else finetuner
        self.evaluator = dump(evaluator) if not isinstance(evaluator, str) else evaluator
        if not isinstance(dummy_input, str):
            buff = io.BytesIO()
            torch.save(dummy_input, buff)
            buff.seek(0)
            dummy_input = base64.b64encode(buff.read()).decode()
        self.dummy_input = dummy_input
        self.trainer = dump(trainer) if not isinstance(trainer, str) else trainer
        if not isinstance(optimizer_helper, str):
            if not isinstance(optimizer_helper, OptimizerConstructHelper):
                optimizer_helper = OptimizerConstructHelper.from_trace(model, optimizer_helper)
            optimizer_helper = dump(optimizer_helper)
        self.optimizer_helper = optimizer_helper
        self.criterion = dump(criterion) if not isinstance(criterion, str) else criterion
        self.device = str(device)

    def export(self) -> Tuple[Module, Callable[[Module], None], Callable[[Module], float], Tensor,
                              Optional[Callable[[Module, Optimizer, Callable[[Any, Any], Any]], None]],
                              Optional[OptimizerConstructHelper], Optional[Callable[[Any, Any], Any]], torch.device]:
        device = torch.device(self.device)
        model = load(self.model)
        if Path('nni_outputs', 'checkpoint', 'model_state_dict.pth').exists():
            model.load_state_dict(torch.load(Path('nni_outputs', 'checkpoint', 'model_state_dict.pth')))
        return (
            model.to(device),
            load(self.finetuner),
            load(self.evaluator),
            torch.load(io.BytesIO(base64.b64decode(self.dummy_input.encode()))).to(device),
            load(self.trainer),
            load(self.optimizer_helper),
            load(self.criterion),
            device
        )

    def json(self):
        canon = self.canonical_copy()
        return asdict(canon)
