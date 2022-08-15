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
import itertools
from typing import Any, Type

import torch.nn as nn

from nni.nas.execution.common import Model
from nni.nas.strategy.base import BaseStrategy
from nni.nas.evaluator.pytorch.lightning import Lightning, LightningModule
from nni.nas.utils.misc import STATE_DICT_PY_MAPPING, STATE_DICT_PY_MAPPING_PARTIAL

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
        """The behavior of export top models in strategy depends on the implementation of inner one-shot module."""
        if self.model is None:
            raise RuntimeError('One-shot strategy needs to be run before export.')
        if top_k != 1:
            warnings.warn('One-shot strategy currently only supports exporting top-1 model.', RuntimeWarning)
        return [self.model.export()]

    @staticmethod
    def sub_state_dict(model, super_state_dict):
        """
        Truncate the state dict of the fixed subnet from that of the search space.

        For example, when you already have a state dict for the base model / search space (which often
        happens when you have trained a supernet with one-shot strategies), the state dict isn't organized
        in the same way as when a sub-model is sampled from the search space. This patch will help
        the modules in the sub-model find the corresponding name and param in the base model.
        """
        assert isinstance(model, nn.Module), 'PyTorch is the only supported framework for now.'
        # first get the full mapping
        full_mapping = {}
        state_dict = model.state_dict()
        def update_state_dict(src_prefix, tar_prefix, module):
            if hasattr(module, STATE_DICT_PY_MAPPING):
                # only values are complete
                local_map = getattr(module, STATE_DICT_PY_MAPPING)
            elif hasattr(module, STATE_DICT_PY_MAPPING_PARTIAL):
                # keys and values are both incomplete
                local_map = getattr(module, STATE_DICT_PY_MAPPING_PARTIAL)
                local_map = {k: tar_prefix + v for k, v in local_map.items()}
            else:
                # no mapping
                local_map = {}

            if '__self__' in local_map:
                # special case, overwrite prefix
                tar_prefix = local_map['__self__'] + '.'

            for key, value in local_map.items():
                if key != '' and key not in module._modules:  # not a sub-module, probably a parameter
                    full_mapping[src_prefix + key] = value

            # To deal with leaf nodes.
            for name, sub_param in itertools.chain(module._parameters.items(), module._buffers.items()):  # direct children
                if sub_param is None or name in module._non_persistent_buffers_set:
                    # it won't appear in state dict
                    continue
                if (src_prefix + name) not in full_mapping:
                    full_mapping[src_prefix + name] = tar_prefix + name

                sup_param = super_state_dict[tar_prefix + name]

                assert sub_param.ndim == sup_param.ndim
                # truncation operation of custom defined mixed module
                if hasattr(module, "truncation"):
                    indices = getattr(module, "truncation")(name, sub_param.shape, sup_param.shape)
                # conv2d weight truncation
                # TODO: groups>1 unsupported, fixe me later.
                elif isinstance(module, nn.Conv2d) and name == "weight":
                    indices = [slice(0, min(i, j)) for i, j in zip(sub_param.shape[:2], sup_param.shape[:2])]
                    kernel_a, kernel_b = module.kernel_size
                    max_kernel_a, max_kernel_b = sup_param.shape[2:]
                    kernel_a_left, kernel_b_top = (max_kernel_a - kernel_a) // 2, (max_kernel_b - kernel_b) // 2
                    indices.extend([slice(kernel_a_left, kernel_a_left + kernel_a), slice(kernel_b_top, kernel_b_top + kernel_b)])
                # default head truncation
                else:
                    indices = [slice(0, min(i, j)) for i, j in zip(sub_param.shape, sup_param.shape)]

                state_dict[src_prefix + name] = sup_param[indices]

            for name, child in module.named_children():
                # sub-modules
                update_state_dict(
                    src_prefix + name + '.',
                    local_map.get(name, tar_prefix + name) + '.',  # if mapping doesn't exist, respect the prefix
                    child
                )

        update_state_dict('', '', model)

        return state_dict


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
