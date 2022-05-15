# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import warnings
from typing import cast

import tensorflow as tf

from nni.retiarii.utils import ModelNamespace


class LayerChoice(tf.keras.Model):
    # FIXME: This is only a draft to test multi-framework support, not properly tested at all.

    def __init__(self, candidates: dict[str, tf.Module] | list[tf.Module], *,
                 prior: list[float] | None = None, label: str | None = None, **kwargs):
        super().__init__()
        self.candidates = candidates
        self.prior = prior or [1 / len(candidates) for _ in range(len(candidates))]
        assert abs(sum(self.prior) - 1) < 1e-5, 'Sum of prior distribution is not 1.'
        self._label = generate_new_label(label)

        self.names = []
        if isinstance(candidates, dict):
            for name, module in candidates.items():
                assert name not in ["length", "reduction", "return_mask", "_key", "key", "names"], \
                    "Please don't use a reserved name '{}' for your module.".format(name)
                self.add_module(name, module)
                self.names.append(name)
        elif isinstance(candidates, list):
            for i, module in enumerate(candidates):
                self.add_module(str(i), module)
                self.names.append(str(i))
        else:
            raise TypeError("Unsupported candidates type: {}".format(type(candidates)))
        self._first_module = cast(tf.Module, self._modules[self.names[0]])  # to make the dummy forward meaningful

    @property
    def label(self):
        return self._label

    def __getitem__(self, idx: int | str) -> tf.Module:
        if isinstance(idx, str):
            return cast(tf.Module, self._modules[idx])
        return cast(tf.Module, list(self)[idx])

    def __setitem__(self, idx, module):
        key = idx if isinstance(idx, str) else self.names[idx]
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in self.names[idx]:
                delattr(self, key)
        else:
            if isinstance(idx, str):
                key, idx = idx, self.names.index(idx)
            else:
                key = self.names[idx]
            delattr(self, key)
        del self.names[idx]

    def __len__(self):
        return len(self.names)

    def __iter__(self):
        return map(lambda name: self._modules[name], self.names)

    def call(self, x):
        """
        The forward of layer choice is simply running the first candidate module.
        It shouldn't be called directly by users in most cases.
        """
        warnings.warn('You should not run forward of this module directly.')
        return self._first_module(x)

    def __repr__(self):
        return f'LayerChoice({self.candidates}, label={repr(self.label)})'


def generate_new_label(label: str | None):
    if label is None:
        return ModelNamespace.next_label()
    return label
