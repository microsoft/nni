from typing import Type

import pytest

import torch
from nni.nas.space import ExecutableModelSpace, GraphModelSpace

from .models import (
    CellSimple, CellDefaultArgs, CellCustomProcessor, CellLooseEnd, CellOpFactory
)

from .test_choice import space_format


@pytest.fixture
def skip_if_graph(space_format):
    if issubclass(space_format, GraphModelSpace):
        pytest.skip("GraphSpace does not support cell")


def test_cell(space_format: Type[ExecutableModelSpace], skip_if_graph):
    net = CellSimple()
    assert net.cell.label == 'model/1'
    assert net.cell.ops[0][0].label == 'model/1/op_2_0'
    model = space_format.from_model(net)
    for _ in range(10):
        selected_model = model.random().executable_model()
        assert selected_model(torch.randn(1, 16), torch.randn(1, 16)).size() == torch.Size([1, 64])

    model = space_format.from_model(CellDefaultArgs())
    for _ in range(10):
        selected_model = model.random().executable_model()
        assert selected_model(torch.randn(1, 16)).size() == torch.Size([1, 64])


def test_cell_predecessors(space_format: Type[ExecutableModelSpace], skip_if_graph):
    model = space_format.from_model(CellCustomProcessor())
    for _ in range(10):
        selected_model = model.random().executable_model()
        result = selected_model(torch.randn(1, 3), torch.randn(1, 16))
        assert result[0].size() == torch.Size([1, 16])
        assert result[1].size() == torch.Size([1, 64])


def test_cell_loose_end(space_format: Type[ExecutableModelSpace], skip_if_graph):
    model = space_format.from_model(CellLooseEnd())
    any_not_all = False
    for _ in range(10):
        selected_model = model.random().executable_model()
        indices = selected_model.cell.output_node_indices
        assert all(i >= 2 for i in indices)
        assert selected_model(torch.randn(1, 16), torch.randn(1, 16)).size() == torch.Size([1, 16 * len(indices)])
        if len(indices) < 4:
            any_not_all = True
    assert any_not_all


def test_cell_complex(space_format: Type[ExecutableModelSpace], skip_if_graph):
    model = space_format.from_model(CellOpFactory())
    for _ in range(10):
        selected_model = model.random().executable_model()
        assert selected_model(
            torch.randn(1, 3), torch.randn(1, 16)).size() == torch.Size([1, 64])
