"""Strategy integration of one-shot.

This file is put here simply because it relies on "pytorch".
For consistency, please consider importing strategies from ``nni.retiarii.strategy``.
For example, ``nni.retiarii.strategy.DartsStrategy`` (this requires pytorch to be installed of course).
"""

import warnings
from typing import Any, List, Optional, Type

import torch.nn as nn
from nni.retiarii.graph import Evaluator, Model
from nni.retiarii.strategy.base import BaseStrategy
from nni.retiarii.evaluator.pytorch.lightning import Lightning
from torch.utils.data import DataLoader

from .base_lightning import BaseOneShotLightningModule
from .differentiable import DartsModule
from .utils import InterleavedTrainValDataLoader


class OneShotStrategy(BaseStrategy):

    def __init__(self, oneshot_module: Type[BaseOneShotLightningModule], **kwargs):
        self.oneshot_module = oneshot_module
        self.oneshot_kwargs = kwargs

        self.model: Optional[BaseOneShotLightningModule] = None

    def get_dataloader(self, train_dataloader: DataLoader, val_dataloaders: DataLoader):
        """
        One-shot strategy typically requires a customized dataloader.
        """
        raise NotImplementedError()

    def run(self, base_model: Model, applied_mutators):
        # one-shot strategy doesn't use ``applied_mutators``
        # but get the "mutators" on their own

        _reason = 'The reason might be that you have used the wrong execution engine. Try to set engine to `oneshot` and try again.'

        py_model: nn.Module = base_model.python_object
        if not isinstance(py_model, nn.Module):
            raise TypeError('Model is not a nn.Module. ' + _reason)

        if applied_mutators:
            raise ValueError('Mutator is not empty. ' + _reason)
        
        if not isinstance(base_model.evaluator, Lightning):
            raise TypeError('Evaluator needs to be a lightning evaluator to make one-shot strategy work.')

        self.model: BaseOneShotLightningModule = self.oneshot_module(py_model, **self.oneshot_kwargs)
        evaluator: Lightning = base_model.evaluator
        dataloader = self.get_dataloader(evaluator.train_dataloader, evaluator.val_dataloaders)
        evaluator.trainer.fit(py_model, dataloader)

    def export_top_models(self, top_k: int = 1) -> List[Any]:
        if self.model is not None:
            raise RuntimeError('One-shot strategy needs to be run before export.')
        if top_k != 1:
            warnings.warn('One-shot strategy currently only supports exporting top-1 model.', RuntimeWarning)
        return [self.model.export()]


class DartsStrategy(OneShotStrategy):
    __doc__ = DartsModule._darts_note.format(module_notes='', module_params='')

    def __init__(self, **kwargs):
        super().__init__(DartsModule, **kwargs)

    def get_dataloader(self, train_dataloader: DataLoader, val_dataloaders: DataLoader):
        return InterleavedTrainValDataLoader(train_dataloader, val_dataloaders)
