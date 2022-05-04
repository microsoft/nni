import base64
import io
from dataclasses import dataclass, asdict
from typing import Any, Dict, Callable, Optional, Tuple, Union
from typing_extensions import Literal

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from nni.algorithms.compression.v2.pytorch.utils.constructor_helper import OptimizerConstructHelper
from nni.common import dump, load
from nni.experiment.config.base import ConfigBase


@dataclass(init=False)
class CompressionVessel(ConfigBase):
    model: Union[str, bytes]
    finetuner: Union[str, bytes]
    evaluator: Union[str, bytes]
    dummy_input: Union[str, bytes]
    trainer: Union[str, bytes, Literal['null'], None]
    optimizer_helper: Union[str, bytes, Literal['null'], None]
    criterion: Union[str, bytes, Literal['null'], None]
    device: str

    def __init__(self,
                 model: Module,
                 finetuner: Callable[[Module], None],
                 evaluator: Callable[[Module], float],
                 dummy_input: Tensor,
                 trainer: Optional[Callable[[Module, Optimizer, Callable[[Any, Any], Any]], None]],
                 optimizer_helper: Union[Optimizer, OptimizerConstructHelper, None],
                 criterion: Optional[Callable[[Any, Any], Any]],
                 device: str):
        # should be wrote as override __init__
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
        return (
            load(self.model).to(device),
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
