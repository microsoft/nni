# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from nni.retiarii.nn.pytorch import Cell

from .base import BaseSuperNetModule


class SupernetCell(BaseSuperNetModule):
    """The implementation of super-net cell follows `DARTS <https://github.com/quark0/darts>`__.

    When ``factory_used`` is true, it reconstructs the cell for every possible combination of operation and input index.
    On export, we first have best (operation, input) pairs, the select the best ``num_ops_per_node``.

    ``loose_end`` is not supported yet, because it will cause more problems (e.g., shape mismatch).
    We assumes ``loose_end`` to be ``all`` regardless of its configuration.

    A supernet cell can't slim its own weight to fit into a sub network, which is also a known issue.
    """

    @classmethod
    def mutate(cls, module, name, memo, mutate_kwargs):
        if isinstance(module, Cell):
            size = len(module)
            if module.label in memo:
                alpha = memo[module.label]
                if len(alpha) != size:
                    raise ValueError(f'Architecture parameter size of same label {module.label} conflict: {len(alpha)} vs. {size}')
            else:
                alpha = nn.Parameter(torch.randn(size) * 1E-3)  # this can be reinitialized later

            softmax = mutate_kwargs.get('softmax', nn.Softmax(-1))
            return cls(list(module.named_children()), alpha, softmax, module.label)

