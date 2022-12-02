# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Strategy integration of one-shot.

This file is put here simply because it relies on "pytorch".
For consistency, please consider importing strategies from ``nni.nas.strategy``.
For example, ``nni.nas.strategy.DartsStrategy`` (this requires pytorch to be installed of course).

When adding/modifying a new strategy in this file, don't forget to link it in strategy/oneshot.py.
"""

from __future__ import annotations

import warnings
from typing import Any, Type, Union

import torch.nn as nn

from nni.nas.execution.common import Model
from nni.nas.strategy.base import BaseStrategy
from nni.nas.evaluator.pytorch.lightning import Lightning, LightningModule

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

    def attach_model(self, base_model: Union[Model, nn.Module]):
        _reason = 'The reason might be that you have used the wrong execution engine. Try to set engine to `oneshot` and try again.'

        if isinstance(base_model, Model):
            if not isinstance(base_model.python_object, nn.Module):
                raise TypeError('Model is not a nn.Module. ' + _reason)
            py_model: nn.Module = base_model.python_object
            if not isinstance(base_model.evaluator, Lightning):
                raise TypeError('Evaluator needs to be a lightning evaluator to make one-shot strategy work.')
            evaluator_module: LightningModule = base_model.evaluator.module
            evaluator_module.running_mode = 'oneshot'
            evaluator_module.set_model(py_model)
        else:
            # FIXME: this should be an evaluator + model
            from nni.retiarii.evaluator.pytorch.lightning import ClassificationModule
            evaluator_module = ClassificationModule(num_classes=10)
            evaluator_module.running_mode = 'oneshot'
            evaluator_module.set_model(base_model)
        self.model = self.oneshot_module(evaluator_module, **self.oneshot_kwargs)

    def run(self, base_model: Model, applied_mutators):
        # one-shot strategy doesn't use ``applied_mutators``
        # but get the "mutators" on their own

        _reason = 'The reason might be that you have used the wrong execution engine. Try to set engine to `oneshot` and try again.'

        if applied_mutators:
            raise ValueError('Mutator is not empty. ' + _reason)

        if not isinstance(base_model.evaluator, Lightning):
            raise TypeError('Evaluator needs to be a lightning evaluator to make one-shot strategy work.')

        self.attach_model(base_model)
        evaluator: Lightning = base_model.evaluator
        if evaluator.train_dataloaders is None or evaluator.val_dataloaders is None:
            raise TypeError('Training and validation dataloader are both required to set in evaluator for one-shot strategy.')
        train_loader, val_loader = self.preprocess_dataloader(evaluator.train_dataloaders, evaluator.val_dataloaders)
        assert isinstance(self.model, BaseOneShotLightningModule)
        evaluator.trainer.fit(self.model, train_loader, val_loader)

    def export_top_models(self, top_k: int = 1) -> list[Any]:
        """The behavior of export top models in strategy depends on the implementation of inner one-shot module."""
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

    def sub_state_dict(self, arch: dict[str, Any]):
        """Export the state dict of a chosen architecture.
        This is useful in weight inheritance of subnet as was done in
        `SPOS <https://arxiv.org/abs/1904.00420>`__,
        `OFA <https://arxiv.org/abs/1908.09791>`__ and
        `AutoFormer <https://arxiv.org/abs/2106.13008>`__.

        Parameters
        ----------
        arch
            The architecture to be exported.

        Examples
        --------
        To obtain a state dict of a chosen architecture, you can use the following code::

            # Train or load a random one-shot strategy
            experiment.run(...)
            best_arch = experiment.export_top_models()[0]

            # If users are to manipulate checkpoint in an evaluator,
            # they should use this `no_fixed_arch()` statement to make sure
            # instantiating model space works properly, as evaluator is running in a fixed context.
            from nni.nas.fixed import no_fixed_arch
            with no_fixed_arch():
                model_space = MyModelSpace()    # must create a model space again here

            # If the strategy has been created previously, directly use it.
            strategy = experiment.strategy

            # Or load a strategy from a checkpoint
            strategy = RandomOneShot()
            strategy.attach_model(model_space)
            strategy.model.load_state_dict(torch.load(...))

            state_dict = strategy.sub_state_dict(best_arch)

        The state dict can be directly loaded into a fixed architecture using ``fixed_arch``::

            with fixed_arch(best_arch):
                model = MyModelSpace()
            model.load_state_dict(state_dict)

        Another common use case is to search for a subnet on supernet with a multi-trial strategy (e.g., evolution).
        The key step here is to write a customized evaluator that loads the checkpoint from the supernet and run evaluations::

            def evaluate_model(model_fn):
                model = model_fn()

                # Put this into `on_validation_start` or `on_train_start` if using Lightning evaluator.
                model.load_state_dict(get_subnet_state_dict())
                # Batch-norm calibration is often needed for better performance,
                # which is often running several hundreds of mini-batches to
                # re-compute running statistics of batch normalization for subnets.
                # See https://arxiv.org/abs/1904.00420 for details.
                finetune_bn(model)
                # Alternatively, you can also set batch norm to train mode to disable running statistics.
                # model.train()

                # Evaluate the model and validation dataloader.
                evaluate_acc(model)

        ``get_subnet_state_dict()`` here is a bit tricky. It's mostly the same as the pervious use case,
        but the architecture dict should be obtained from ``mutation_summary`` in ``get_current_parameter()``,
        which corresponds to the architecture of the current trial::

            def get_subnet_state_dict():
                random_oneshot_strategy = load_random_oneshot_strategy()     # Load a strategy from checkpoint, same as above
                arch_dict = nni.get_current_parameter()['mutation_summary']
                print('Architecture dict:', arch_dict)                       # Print here to see what it looks like
                return random_oneshot_strategy.sub_state_dict(arch_dict)
        """
        assert isinstance(self.model, RandomSamplingLightningModule)
        return self.model.sub_state_dict(arch)
