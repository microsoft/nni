import json

import torch

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

    def _encode_tensor(self, data):
        if isinstance(data, list):
            if all(map(lambda o: isinstance(o, bool), data)):
                return torch.tensor(data, dtype=torch.bool)  # pylint: disable=not-callable
            else:
                return torch.tensor(data, dtype=torch.float)  # pylint: disable=not-callable
        if isinstance(data, dict):
            return {k: self._encode_tensor(v) for k, v in data.items()}
        return data

    def before_pass(self):
        self._unused_key = set(self._fixed_arc.keys())

    def after_pass(self):
        if self._strict:
            if self._unused_key:
                raise ValueError("{} are never used by the network. "
                                 "Set strict=False if you want to disable this check.".format(self._unused_key))

    def _check_key(self, key):
        if key not in self._fixed_arc:
            raise ValueError("\"{}\" is demanded by the network, but not found in saved architecture.".format(key))
        if key in self._unused_key:
            self._unused_key.remove(key)

    def on_calc_layer_choice_mask(self, mutable):
        self._check_key(mutable.key)
        return self._fixed_arc[mutable.key]

    def on_calc_input_choice_mask(self, mutable, tags):
        self._check_key(mutable.key)
        return self._fixed_arc[mutable.key]
