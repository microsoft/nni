# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Strategy integration of one-shot.

This file is put here simply because it relies on "pytorch".
For consistency, please consider importing strategies from ``nni.retiarii.strategy``.
For example, ``nni.retiarii.strategy.DartsStrategy`` (this requires pytorch to be installed of course).

When adding/modifying a new strategy in this file, don't forget to link it in strategy/oneshot.py.
"""

from __future__ import annotations

import warnings
from typing import Any, Type

import torch.nn as nn

from nni.retiarii.graph import Model
from nni.retiarii.strategy.base import BaseStrategy
from nni.retiarii.evaluator.pytorch.lightning import Lightning, LightningModule

from .base_lightning import BaseOneShotLightningModule
from .differentiable import DartsLightningModule, ProxylessLightningModule, GumbelDartsLightningModule
from .sampling import EnasLightningModule, RandomSamplingLightningModule


class OneShotStrategy(BaseStrategy):
    """Wrap an one-shot lightning module as a one-shot strategy."""

    def __init__(self, oneshot_module: Type[BaseOneShotLightningModule], **kwargs):
        self.oneshot_module = oneshot_module
        self.oneshot_kwargs = kwargs

        self.model: BaseOneShotLightningModule | None = None

    def preprocess_dataloader(self, train_dataloaders: Any, val_dataloaders: Any) -> tuple[Any, Any]:
        """
        One-shot strategy typically requires fusing train and validation dataloader in an ad-hoc way.
        As one-shot strategy doesn't try to open the blackbox of a batch,
        theoretically, these dataloader can be
        `any dataloader types supported by Lightning <https://pytorch-lightning.readthedocs.io/en/stable/guides/data.html>`__.

        Returns
        -------
        A tuple of preprocessed train dataloaders and validation dataloaders.
        """
        return train_dataloaders, val_dataloaders

    def run(self, base_model: Model, applied_mutators):
        # one-shot strategy doesn't use ``applied_mutators``
        # but get the "mutators" on their own

        _reason = 'The reason might be that you have used the wrong execution engine. Try to set engine to `oneshot` and try again.'

        if not isinstance(base_model.python_object, nn.Module):
            raise TypeError('Model is not a nn.Module. ' + _reason)
        py_model: nn.Module = base_model.python_object

        if applied_mutators:
            raise ValueError('Mutator is not empty. ' + _reason)

        if not isinstance(base_model.evaluator, Lightning):
            raise TypeError('Evaluator needs to be a lightning evaluator to make one-shot strategy work.')

        evaluator_module: LightningModule = base_model.evaluator.module
        evaluator_module.running_mode = 'oneshot'
        evaluator_module.set_model(py_model)

        self.model = self.oneshot_module(evaluator_module, **self.oneshot_kwargs)
        evaluator: Lightning = base_model.evaluator
        if evaluator.train_dataloaders is None or evaluator.val_dataloaders is None:
            raise TypeError('Training and validation dataloader are both required to set in evaluator for one-shot strategy.')
        train_loader, val_loader = self.preprocess_dataloader(evaluator.train_dataloaders, evaluator.val_dataloaders)
        evaluator.trainer.fit(self.model, train_loader, val_loader)

    def export_top_models(self, top_k: int = 1) -> list[Any]:
        if self.model is None:
            raise RuntimeError('One-shot strategy needs to be run before export.')
        if top_k != 1:
            warnings.warn('One-shot strategy currently only supports exporting top-1 model.', RuntimeWarning)
        return [self.model.export()]


class DARTS(OneShotStrategy):
    __doc__ = DartsLightningModule._darts_note.format(module_notes='', module_params='')

    def __init__(self, **kwargs):
        super().__init__(DartsLightningModule, **kwargs)

    def preprocess_dataloader(self, train_dataloaders, val_dataloaders):
        # By returning a dict, we make a CombinedLoader (in Lightning)
        return {
            'train': train_dataloaders,
            'val': val_dataloaders
        }, None


class Proxyless(OneShotStrategy):
    __doc__ = ProxylessLightningModule._proxyless_note.format(module_notes='', module_params='')

    def __init__(self, **kwargs):
        super().__init__(ProxylessLightningModule, **kwargs)

    def preprocess_dataloader(self, train_dataloaders, val_dataloaders):
        return {
            'train': train_dataloaders,
            'val': val_dataloaders
        }, None


class GumbelDARTS(OneShotStrategy):
    __doc__ = GumbelDartsLightningModule._gumbel_darts_note.format(module_notes='', module_params='')

    def __init__(self, **kwargs):
        super().__init__(GumbelDartsLightningModule, **kwargs)

    def preprocess_dataloader(self, train_dataloaders, val_dataloaders):
        return {
            'train': train_dataloaders,
            'val': val_dataloaders
        }, None


class ENAS(OneShotStrategy):
    __doc__ = EnasLightningModule._enas_note.format(module_notes='', module_params='')

    def __init__(self, **kwargs):
        super().__init__(EnasLightningModule, **kwargs)

    def preprocess_dataloader(self, train_dataloaders, val_dataloaders):
        # Import locally to avoid import error on legacy PL version
        from .dataloader import ConcatLoader
        return ConcatLoader({
            'train': train_dataloaders,
            'val': val_dataloaders
        }), None


class RandomOneShot(OneShotStrategy):
    __doc__ = RandomSamplingLightningModule._random_note.format(module_notes='', module_params='')

    def __init__(self, **kwargs):
        super().__init__(RandomSamplingLightningModule, **kwargs)
