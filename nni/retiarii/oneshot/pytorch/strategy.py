# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Strategy integration of one-shot.

This file is put here simply because it relies on "pytorch".
For consistency, please consider importing strategies from ``nni.retiarii.strategy``.
For example, ``nni.retiarii.strategy.DartsStrategy`` (this requires pytorch to be installed of course).

When adding/modifying a new strategy in this file, don't forget to link it in strategy/oneshot.py.
"""

import warnings
from typing import Any, List, Optional, Type, Union, Tuple

import torch.nn as nn
from torch.utils.data import DataLoader

from nni.retiarii.graph import Model
from nni.retiarii.strategy.base import BaseStrategy
from nni.retiarii.evaluator.pytorch.lightning import Lightning, LightningModule

from .base_lightning import BaseOneShotLightningModule
from .differentiable import DartsLightningModule, ProxylessLightningModule, GumbelDartsLightningModule
from .sampling import EnasLightningModule, RandomSamplingLightningModule
from .utils import InterleavedTrainValDataLoader, ConcatenateTrainValDataLoader


class OneShotStrategy(BaseStrategy):
    """Wrap an one-shot lightning module as a one-shot strategy."""

    def __init__(self, oneshot_module: Type[BaseOneShotLightningModule], **kwargs):
        self.oneshot_module = oneshot_module
        self.oneshot_kwargs = kwargs

        self.model: Optional[BaseOneShotLightningModule] = None

    def _get_dataloader(self, train_dataloader: DataLoader, val_dataloaders: DataLoader) \
        -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
        """
        One-shot strategy typically requires a customized dataloader.

        If only train dataloader is produced, return one dataloader.
        Otherwise, return train dataloader and valid loader as a tuple.
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

        evaluator_module: LightningModule = base_model.evaluator.module
        evaluator_module.set_model(py_model)

        self.model: BaseOneShotLightningModule = self.oneshot_module(evaluator_module, **self.oneshot_kwargs)
        evaluator: Lightning = base_model.evaluator
        dataloader = self._get_dataloader(evaluator.train_dataloader, evaluator.val_dataloaders)
        if isinstance(dataloader, tuple):
            dataloader, val_loader = dataloader
            evaluator.trainer.fit(self.model, dataloader, val_loader)
        else:
            evaluator.trainer.fit(self.model, dataloader)

    def export_top_models(self, top_k: int = 1) -> List[Any]:
        if self.model is None:
            raise RuntimeError('One-shot strategy needs to be run before export.')
        if top_k != 1:
            warnings.warn('One-shot strategy currently only supports exporting top-1 model.', RuntimeWarning)
        return [self.model.export()]


class DARTS(OneShotStrategy):
    __doc__ = DartsLightningModule._darts_note.format(module_notes='', module_params='')

    def __init__(self, **kwargs):
        super().__init__(DartsLightningModule, **kwargs)

    def _get_dataloader(self, train_dataloader, val_dataloaders):
        return InterleavedTrainValDataLoader(train_dataloader, val_dataloaders)


class Proxyless(OneShotStrategy):
    __doc__ = ProxylessLightningModule._proxyless_note.format(module_notes='', module_params='')

    def __init__(self, **kwargs):
        super().__init__(ProxylessLightningModule, **kwargs)

    def _get_dataloader(self, train_dataloader, val_dataloaders):
        return InterleavedTrainValDataLoader(train_dataloader, val_dataloaders)


class GumbelDARTS(OneShotStrategy):
    __doc__ = GumbelDartsLightningModule._gumbel_darts_note.format(module_notes='', module_params='')

    def __init__(self, **kwargs):
        super().__init__(GumbelDartsLightningModule, **kwargs)

    def _get_dataloader(self, train_dataloader, val_dataloaders):
        return InterleavedTrainValDataLoader(train_dataloader, val_dataloaders)


class ENAS(OneShotStrategy):
    __doc__ = EnasLightningModule._enas_note.format(module_notes='', module_params='')

    def __init__(self, **kwargs):
        super().__init__(EnasLightningModule, **kwargs)

    def _get_dataloader(self, train_dataloader, val_dataloaders):
        return ConcatenateTrainValDataLoader(train_dataloader, val_dataloaders)


class RandomOneShot(OneShotStrategy):
    __doc__ = RandomSamplingLightningModule._random_note.format(module_notes='', module_params='')

    def __init__(self, **kwargs):
        super().__init__(RandomSamplingLightningModule, **kwargs)

    def _get_dataloader(self, train_dataloader, val_dataloaders):
        return train_dataloader, val_dataloaders
