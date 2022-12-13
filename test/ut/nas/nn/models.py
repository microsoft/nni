from typing import List, Tuple

import torch
import torch.nn as nn

from nni.nas.nn.pytorch import ModelSpace, Cell, ParametrizedModule


class CellSimple(ModelSpace):
    def __init__(self):
        super().__init__()
        self.cell = Cell([nn.Linear(16, 16), nn.Linear(16, 16, bias=False)],
                         num_nodes=4, num_ops_per_node=2, num_predecessors=2, merge_op='all')

    def forward(self, x, y):
        return self.cell(x, y)


class CellDefaultArgs(ModelSpace):
    def __init__(self):
        super().__init__()
        self.cell = Cell([nn.Linear(16, 16), nn.Linear(16, 16, bias=False)], num_nodes=4)

    def forward(self, x):
        return self.cell(x)


class CellPreprocessor(ModelSpace):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 16)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        return [self.linear(x[0]), x[1]]


class CellPostprocessor(nn.Module):
    def forward(self, this: torch.Tensor, prev: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return prev[-1], this


class CellCustomProcessor(ModelSpace):
    def __init__(self):
        super().__init__()
        self.cell = Cell({
            'first': nn.Linear(16, 16),
            'second': nn.Linear(16, 16, bias=False)
        }, num_nodes=4, num_ops_per_node=2, num_predecessors=2,
            preprocessor=CellPreprocessor(), postprocessor=CellPostprocessor(), merge_op='all')

    def forward(self, x, y):
        return self.cell([x, y])


class CellLooseEnd(ModelSpace):
    def __init__(self):
        super().__init__()
        self.cell = Cell([nn.Linear(16, 16), nn.Linear(16, 16, bias=False)],
                         num_nodes=4, num_ops_per_node=2, num_predecessors=2, merge_op='loose_end')

    def forward(self, x, y):
        return self.cell([x, y])


class CellOpFactory(ModelSpace):
    def __init__(self):
        super().__init__()
        self.cell = Cell({
            'first': lambda _, __, chosen: nn.Linear(3 if chosen == 0 else 16, 16),
            'second': lambda _, __, chosen: nn.Linear(3 if chosen == 0 else 16, 16, bias=False)
        }, num_nodes=4, num_ops_per_node=2, num_predecessors=2, merge_op='all')

    def forward(self, x, y):
        return self.cell([x, y])


class MutableConv(ParametrizedModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=1)
        self.conv2 = nn.Conv2d(3, 5, kernel_size=1)

    def forward(self, x: torch.Tensor, index: int):
        if index == 0:
            return self.conv1(x)
        else:
            return self.conv2(x)
