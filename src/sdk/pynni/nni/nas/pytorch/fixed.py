# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json

import torch

from nni.nas.pytorch.mutables import MutableScope
from nni.nas.pytorch.mutator import Mutator


class FixedArchitecture(Mutator):

    def __init__(self, model, fixed_arc, strict=True):
        """
        Initialize a fixed architecture mutator.

        Parameters
        ----------
        model : nn.Module
            A mutable network.
        fixed_arc : str or dict
            Path to the architecture checkpoint (a string), or preloaded architecture object (a dict).
        strict : bool
            Force everything that appears in `fixed_arc` to be used at least once.
        """
        super().__init__(model)
        self._fixed_arc = fixed_arc

        mutable_keys = set([mutable.key for mutable in self.mutables if not isinstance(mutable, MutableScope)])
        fixed_arc_keys = set(self._fixed_arc.keys())
        if fixed_arc_keys - mutable_keys:
            raise RuntimeError("Unexpected keys found in fixed architecture: {}.".format(fixed_arc_keys - mutable_keys))
        if mutable_keys - fixed_arc_keys:
            raise RuntimeError("Missing keys in fixed architecture: {}.".format(mutable_keys - fixed_arc_keys))

    def sample_search(self):
        return self._fixed_arc

    def sample_final(self):
        return self._fixed_arc


def _encode_tensor(data, device):
    if isinstance(data, list):
        if all(map(lambda o: isinstance(o, bool), data)):
            return torch.tensor(data, dtype=torch.bool, device=device)  # pylint: disable=not-callable
        else:
            return torch.tensor(data, dtype=torch.float, device=device)  # pylint: disable=not-callable
    if isinstance(data, dict):
        return {k: _encode_tensor(v, device) for k, v in data.items()}
    return data


def apply_fixed_architecture(model, fixed_arc_path, device=None):
    """
    Load architecture from `fixed_arc_path` and apply to model.

    Parameters
    ----------
    model : torch.nn.Module
        Model with mutables.
    fixed_arc_path : str
        Path to the JSON that stores the architecture.
    device : torch.device
        Architecture weights will be transfered to `device`.

    Returns
    -------
    FixedArchitecture
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(fixed_arc_path, str):
        with open(fixed_arc_path, "r") as f:
            fixed_arc = json.load(f)
    fixed_arc = _encode_tensor(fixed_arc, device)
    architecture = FixedArchitecture(model, fixed_arc)
    architecture.to(device)
    architecture.reset()
    return architecture
