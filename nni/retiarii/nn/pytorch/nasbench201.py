from collections import OrderedDict
from typing import List, Callable, Optional

import torch.nn as nn

from .api import LayerChoice
from .utils import generate_new_label


class NasBench201Cell(nn.Module):

    @staticmethod
    def _make_dict(x):
        if isinstance(x, list):
            return OrderedDict([(str(i), t) for i, t in enumerate(x)])
        return OrderedDict(x)

    def __init__(self, op_candidates: List[Callable[[int, int], nn.Module]],
                 in_features: int, out_features: int, num_tensors: int = 4,
                 label: Optional[str] = None):
        super().__init__()
        self._label = generate_new_label(label)

        self.layers = nn.ModuleList()
        self.in_features = in_features
        self.out_features = out_features
        self.num_tensors = num_tensors

        op_candidates = self._make_dict(op_candidates)

        for tid in range(num_tensors):
            node_ops = nn.ModuleList()
            for j in range(tid):
                inp = in_features if j == 0 else out_features
                op_choices = OrderedDict([(key, cls(inp, out_features))
                                          for key, cls in op_candidates.items()])
                node_ops.append(LayerChoice(op_choices, label=f'{self._label}/{j}_{tid}'))
            self.layers.append(node_ops)

    def forward(self, inputs):
        nodes = [inputs]
        for i in range(1, self.num_tensors):
            node_feature = sum(self.layers[i][k](nodes[k]) for k in range(i))
            nodes.append(node_feature)
        return nodes[-1]
