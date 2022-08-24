# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from collections import OrderedDict
import itertools
from typing import Any, Dict

import torch.nn as nn

from nni.common.hpo_utils import ParameterSpec

__all__ = ['BaseSuperNetModule', 'sub_state_dict']


def sub_state_dict(module: Any, destination: Any=None, prefix: str='', keep_vars: bool=False) -> Dict[str, Any]:
    """Returns a dictionary containing a whole state of the BaseSuperNetModule.

    Both parameters and persistent buffers (e.g. running averages) are
    included. Keys are corresponding parameter and buffer names.
    Parameters and buffers set to ``None`` are not included.

    Parameters
    ----------
    arch : dict[str, Any]
        subnet architecture dict.
    destination (dict, optional):
        If provided, the state of module will be updated into the dict
        and the same object is returned. Otherwise, an ``OrderedDict``
        will be created and returned. Default: ``None``.
    prefix (str, optional):
        a prefix added to parameter and buffer names to compose the keys in state_dict.
        Default: ``''``.
    keep_vars (bool, optional):
        by default the :class:`~torch.Tensor` s returned in the state dict are
        detached from autograd. If it's set to ``True``, detaching will not be performed.
        Default: ``False``.

    Returns
    -------
    dict
        Subnet state dictionary.
    """
    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()

    local_metadata = dict(version=module._version)
    if hasattr(destination, "_metadata"):
        destination._metadata[prefix[:-1]] = local_metadata

    if isinstance(module, BaseSuperNetModule):
        module._save_to_sub_state_dict(destination, prefix, keep_vars)
    else:
        module._save_to_state_dict(destination, prefix, keep_vars)
        for name, m in module._modules.items():
            if m is not None:
                sub_state_dict(m, destination=destination, prefix=prefix + name + '.', keep_vars=keep_vars)

    return destination


class BaseSuperNetModule(nn.Module):
    """
    Mutated module in super-net.
    Usually, the feed-forward of the module itself is undefined.
    It has to be resampled with ``resample()`` so that a specific path is selected.
    (Sometimes, this is not required. For example, differentiable super-net.)

    A super-net module usually corresponds to one sample. But two exceptions:

    * A module can have multiple parameter spec. For example, a convolution-2d can sample kernel size, channels at the same time.
    * Multiple modules can share one parameter spec. For example, multiple layer choices with the same label.

    For value choice compositions, the parameter spec are bounded to the underlying (original) value choices,
    rather than their compositions.
    """

    def resample(self, memo: dict[str, Any]) -> dict[str, Any]:
        """
        Resample the super-net module.

        Parameters
        ----------
        memo : dict[str, Any]
            Used to ensure the consistency of samples with the same label.

        Returns
        -------
        dict
            Sampled result. If nothing new is sampled, it should return an empty dict.
        """
        raise NotImplementedError()

    def export(self, memo: dict[str, Any]) -> dict[str, Any]:
        """
        Export the final architecture within this module.
        It should have the same keys as ``search_space_spec()``.

        Parameters
        ----------
        memo : dict[str, Any]
            Use memo to avoid the same label gets exported multiple times.
        """
        raise NotImplementedError()

    def export_probs(self, memo: dict[str, Any]) -> dict[str, Any]:
        """
        Export the probability / logits of every choice got chosen.

        Parameters
        ----------
        memo : dict[str, Any]
            Use memo to avoid the same label gets exported multiple times.
        """
        raise NotImplementedError()

    def search_space_spec(self) -> dict[str, ParameterSpec]:
        """
        Space specification (sample points).
        Mapping from spec name to ParameterSpec. The names in choices should be in the same format of export.

        For example: ::

            {"layer1": ParameterSpec(values=["conv", "pool"])}
        """
        raise NotImplementedError()

    @classmethod
    def mutate(cls, module: nn.Module, name: str, memo: dict[str, Any], mutate_kwargs: dict[str, Any]) -> \
            'BaseSuperNetModule' | bool | tuple['BaseSuperNetModule', bool]:
        """This is a mutation hook that creates a :class:`BaseSuperNetModule`.
        The method should be implemented in each specific super-net module,
        because they usually have specific rules about what kind of modules to operate on.

        Parameters
        ----------
        module : nn.Module
            The module to be mutated (replaced).
        name : str
            Name of this module. With full prefix. For example, ``module1.block1.conv``.
        memo : dict
            Memo to enable sharing parameters among mutated modules. It should be read and written by
            mutate functions themselves.
        mutate_kwargs : dict
            Algo-related hyper-parameters, and some auxiliary information.

        Returns
        -------
        Union[BaseSuperNetModule, bool, tuple[BaseSuperNetModule, bool]]
            The mutation result, along with an optional boolean flag indicating whether to suppress follow-up mutation hooks.
            See :class:`BaseOneShotLightningModule <nni.retiarii.oneshot.pytorch.base_lightning.BaseOneShotLightningModule>` for details.
        """
        raise NotImplementedError()

    def _save_param_buff_to_state_dict(self, destination, prefix, keep_vars):
        """Save the params and buffers of the current module to state dict."""
        for name, value in itertools.chain(self._parameters.items(), self._buffers.items()):  # direct children
            if value is None or name in self._non_persistent_buffers_set:
                # it won't appear in state dict
                continue
            destination[prefix + name] = value if keep_vars else value.detach()

    def _save_module_to_state_dict(self, destination, prefix, keep_vars):
        """Save the sub-module to state dict."""
        for name, module in self._modules.items():
            if module is not None:
                sub_state_dict(module, destination=destination, prefix=prefix + name + '.', keep_vars=keep_vars)

    def _save_to_sub_state_dict(self, destination, prefix, keep_vars):
        """Save to state dict."""
        self._save_param_buff_to_state_dict(destination, prefix, keep_vars)
        self._save_module_to_state_dict(destination, prefix, keep_vars)
