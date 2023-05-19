# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

__all__ = ['NasBench201Cell']

from collections import OrderedDict
from typing import Callable, List, Dict, Union, Optional

import torch
import torch.nn as nn

from nni.mutable import label_scope
from nni.nas.nn.pytorch import LayerChoice, MutableModule


class NasBench201Cell(MutableModule):
    """
    Cell structure that is proposed in NAS-Bench-201.

    Proposed by `NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search <https://arxiv.org/abs/2001.00326>`__.

    This cell is a densely connected DAG with ``num_tensors`` nodes, where each node is tensor.
    For every :math:`i < j`, there is an edge from i-th node to j-th node.
    Each edge in this DAG is associated with an operation transforming the hidden state from the source node
    to the target node. All possible operations are selected from a predefined operation set, defined in ``op_candidates``.
    Each of the ``op_candidates`` should be a callable that accepts input dimension and output dimension,
    and returns a ``Module``.

    Input of this cell should be of shape :math:`[N, C_{in}, *]`, while output should be :math:`[N, C_{out}, *]`. For example,

    The space size of this cell would be :math:`|op|^{N(N-1)/2}`, where :math:`|op|` is the number of operation candidates,
    and :math:`N` is defined by ``num_tensors``.

    Parameters
    ----------
    op_candidates : list of callable
        Operation candidates. Each should be a function accepts input feature and output feature, returning nn.Module.
    in_features : int
        Input dimension of cell.
    out_features : int
        Output dimension of cell.
    num_tensors : int
        Number of tensors in the cell (input included). Default: 4
    label : str
        Identifier of the cell. Cell sharing the same label will semantically share the same choice.
    """

    @staticmethod
    def _make_dict(x):
        if isinstance(x, list):
            return OrderedDict([(str(i), t) for i, t in enumerate(x)])
        return OrderedDict(x)

    def __init__(self, op_candidates: Union[Dict[str, Callable[[int, int], nn.Module]], List[Callable[[int, int], nn.Module]]],
                 in_features: int, out_features: int, num_tensors: int = 4,
                 label: Optional[str] = None):
        super().__init__()

        self.layers = nn.ModuleList()
        self.in_features = in_features
        self.out_features = out_features
        self.num_tensors = num_tensors

        op_candidates = self._make_dict(op_candidates)

        with label_scope(label) as self._scope:
            for tid in range(1, num_tensors):
                node_ops = nn.ModuleList()
                for j in range(tid):
                    inp = in_features if j == 0 else out_features
                    op_choices = OrderedDict([(key, cls(inp, out_features))
                                              for key, cls in op_candidates.items()])
                    node_ops.append(LayerChoice(op_choices, label=f'{j}_{tid}'))
                self.layers.append(node_ops)

    @torch.jit.unused
    @property
    def label(self) -> str:
        return self._scope.name

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        tensors: List[torch.Tensor] = [inputs]
        for layer in self.layers:
            current_tensor: List[torch.Tensor] = []
            for i, op in enumerate(layer):  # type: ignore
                current_tensor.append(op(tensors[i]))  # type: ignore
            tensors.append(torch.sum(torch.stack(current_tensor), 0))
        return tensors[-1]
