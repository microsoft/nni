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
        model: nn.Module
            A mutable network.
        fixed_arc: str or dict
            Path to the architecture checkpoint (a string), or preloaded architecture object (a dict).
        strict: bool
            Force everything that appears in `fixed_arc` to be used at least once.
        """
        super().__init__(model)

        if isinstance(fixed_arc, str):
            with open(fixed_arc, "r") as f:
                fixed_arc = json.load(f)
        self._fixed_arc = self._encode_tensor(fixed_arc)
        self._strict = strict

        mutable_keys = set([mutable.key for mutable in self.mutables if not isinstance(mutable, MutableScope)])
        fixed_arc_keys = set(self._fixed_arc.keys())
        if fixed_arc_keys - mutable_keys:
            raise RuntimeError("Unexpected keys found in fixed architecture: {}.".format(fixed_arc_keys - mutable_keys))
        if mutable_keys - fixed_arc_keys:
            raise RuntimeError("Missing keys in fixed architecture: {}.".format(mutable_keys - fixed_arc_keys))

    def _encode_tensor(self, data):
        if isinstance(data, list):
            if all(map(lambda o: isinstance(o, bool), data)):
                return torch.tensor(data, dtype=torch.bool)  # pylint: disable=not-callable
            else:
                return torch.tensor(data, dtype=torch.float)  # pylint: disable=not-callable
        if isinstance(data, dict):
            return {k: self._encode_tensor(v) for k, v in data.items()}
        return data

    def sample_search(self):
        return self._fixed_arc

    def sample_final(self):
        return self._fixed_arc
