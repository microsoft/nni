# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Dict, Tuple, Union

import torch.nn as nn

from nni.common.hpo_utils import ParameterSpec


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

    def resample(self, memo: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Resample the super-net module.

        Parameters
        ----------
        memo : Dict[str, Any]
            Used to ensure the consistency of samples with the same label.

        Returns
        -------
        dict
            Sampled result. If nothing new is sampled, it should return an empty dict.
        """
        raise NotImplementedError()

    def export(self, memo: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Export the final architecture within this module.
        It should have the same keys as ``search_space_spec()``.

        Parameters
        ----------
        memo : Dict[str, Any]
            Use memo to avoid the same label gets exported multiple times.
        """
        raise NotImplementedError()

    def search_space_spec(self) -> Dict[str, ParameterSpec]:
        """
        Space specification (sample points).
        Mapping from spec name to ParameterSpec. The names in choices should be in the same format of export.

        For example: ::

            {"layer1": ParameterSpec(values=["conv", "pool"])}
        """
        raise NotImplementedError()

    @classmethod
    def mutate(cls, module: nn.Module, name: str, memo: Dict[str, Any], mutate_kwargs: Dict[str, Any]) -> \
            Union['BaseSuperNetModule', bool, Tuple['BaseSuperNetModule', bool]]:
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
        Union[BaseSuperNetModule, bool, Tuple[BaseSuperNetModule, bool]]
            The mutation result, along with an optional boolean flag indicating whether to suppress follow-up mutation hooks.
            See :class:`nni.retiarii.oneshot.pytorch.base.BaseOneShotLightningModule` for details.
        """
        raise NotImplementedError()
