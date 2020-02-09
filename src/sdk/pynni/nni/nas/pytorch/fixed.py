# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json

import torch

from nni.nas.pytorch.mutables import MutableScope
from nni.nas.pytorch.mutator import Mutator


class FixedArchitecture(Mutator):
    """
    Fixed architecture mutator that always selects a certain graph.

    Parameters
    ----------
    model : nn.Module
        A mutable network.
    fixed_arc : str or dict
        Path to the architecture checkpoint (a string), or preloaded architecture object (a dict).
    strict : bool
        Force everything that appears in ``fixed_arc`` to be used at least once.
    """

    def __init__(self, model, fixed_arc, strict=True):
        super().__init__(model)
        self._fixed_arc = fixed_arc

        mutable_keys = set([mutable.key for mutable in self.mutables if not isinstance(mutable, MutableScope)])
        fixed_arc_keys = set(self._fixed_arc.keys())
        if fixed_arc_keys - mutable_keys:
            raise RuntimeError("Unexpected keys found in fixed architecture: {}.".format(fixed_arc_keys - mutable_keys))
        if mutable_keys - fixed_arc_keys:
            raise RuntimeError("Missing keys in fixed architecture: {}.".format(mutable_keys - fixed_arc_keys))

    def sample_search(self):
        """
        Always returns the fixed architecture.
        """
        return self._fixed_arc

    def sample_final(self):
        """
        Always returns the fixed architecture.
        """
        return self._fixed_arc


def _encode_tensor(data):
    if isinstance(data, list):
        if all(map(lambda o: isinstance(o, bool), data)):
            return torch.tensor(data, dtype=torch.bool)  # pylint: disable=not-callable
        else:
            return torch.tensor(data, dtype=torch.float)  # pylint: disable=not-callable
    if isinstance(data, dict):
        return {k: _encode_tensor(v) for k, v in data.items()}
    return data


def apply_fixed_architecture(model, fixed_arc):
    """
    Load architecture from `fixed_arc` and apply to model.

    Parameters
    ----------
    model : torch.nn.Module
        Model with mutables.
    fixed_arc : str or dict
        Path to the JSON that stores the architecture, or dict that stores the exported architecture.

    Returns
    -------
    FixedArchitecture
        Mutator that is responsible for fixes the graph.
    """

    if isinstance(fixed_arc, str):
        with open(fixed_arc) as f:
            fixed_arc = json.load(f)
    fixed_arc = _encode_tensor(fixed_arc)
    architecture = FixedArchitecture(model, fixed_arc)
    architecture.reset()
    return architecture
