from typing import List, Tuple

import torch
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper


@model_wrapper
class CellSimple(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = nn.Cell([nn.Linear(16, 16), nn.Linear(16, 16, bias=False)],
                            num_nodes=4, num_ops_per_node=2, num_predecessors=2, merge_op='all')

    def forward(self, x, y):
        return self.cell(x, y)

@model_wrapper
class CellDefaultArgs(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = nn.Cell([nn.Linear(16, 16), nn.Linear(16, 16, bias=False)], num_nodes=4)

    def forward(self, x):
        return self.cell(x)


class CellPreprocessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 16)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        return [self.linear(x[0]), x[1]]


class CellPostprocessor(nn.Module):
    def forward(self, this: torch.Tensor, prev: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return prev[-1], this


@model_wrapper
class CellCustomProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = nn.Cell({
            'first': nn.Linear(16, 16),
            'second': nn.Linear(16, 16, bias=False)
        }, num_nodes=4, num_ops_per_node=2, num_predecessors=2,
        preprocessor=CellPreprocessor(), postprocessor=CellPostprocessor(), merge_op='all')

    def forward(self, x, y):
        return self.cell([x, y])


@model_wrapper
class CellLooseEnd(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = nn.Cell([nn.Linear(16, 16), nn.Linear(16, 16, bias=False)],
                            num_nodes=4, num_ops_per_node=2, num_predecessors=2, merge_op='loose_end')

    def forward(self, x, y):
        return self.cell([x, y])


@model_wrapper
class CellOpFactory(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = nn.Cell({
            'first': lambda _, __, chosen: nn.Linear(3 if chosen == 0 else 16, 16),
            'second': lambda _, __, chosen: nn.Linear(3 if chosen == 0 else 16, 16, bias=False)
        }, num_nodes=4, num_ops_per_node=2, num_predecessors=2, merge_op='all')

    def forward(self, x, y):
        return self.cell([x, y])
