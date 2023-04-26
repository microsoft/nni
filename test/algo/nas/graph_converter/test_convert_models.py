import unittest

import torch

import nni.nas.nn.pytorch.layers as nn

from .convert_mixin import ConvertMixin, ConvertWithShapeMixin


class TestModels(unittest.TestCase, ConvertMixin):

    def test_nested_modulelist(self):
        class Net(nn.Module):
            def __init__(self, num_nodes, num_ops_per_node):
                super().__init__()
                self.ops = nn.ModuleList()
                self.num_nodes = num_nodes
                self.num_ops_per_node = num_ops_per_node
                for _ in range(num_nodes):
                    self.ops.append(nn.ModuleList([nn.Linear(16, 16) for __ in range(num_ops_per_node)]))

            def forward(self, x):
                state = x
                for ops in self.ops:
                    for op in ops:
                        state = op(state)
                return state

        model = Net(4, 2)
        x = torch.rand((16, 16), dtype=torch.float)
        self.run_test(model, (x, ))

    def test_append_input_tensor(self):
        from typing import List

        class Net(nn.Module):
            def __init__(self, num_nodes):
                super().__init__()
                self.ops = nn.ModuleList()
                self.num_nodes = num_nodes
                for _ in range(num_nodes):
                    self.ops.append(nn.Linear(16, 16))

            def forward(self, x: List[torch.Tensor]):
                state = x
                for ops in self.ops:
                    state.append(ops(state[-1]))
                return state[-1]

        model = Net(4)
        x = torch.rand((1, 16), dtype=torch.float)
        self.run_test(model, ([x], ), check_value=False)  # FIXME

    def test_channels_shuffle(self):
        class Net(nn.Module):
            def forward(self, x):
                bs, num_channels, height, width = x.size()
                x = x.reshape(bs * num_channels // 2, 2, height * width)
                x = x.permute(1, 0, 2)
                x = x.reshape(2, -1, num_channels // 2, height, width)
                return x[0], x[1]

        model = Net()
        x = torch.rand((1, 64, 224, 224), dtype=torch.float)
        self.run_test(model, (x, ))

    def test_identity_node(self):
        class Net(nn.Module):
            def forward(self, x):
                return x

        model = Net()
        x = torch.rand((1, 64, 224, 224), dtype=torch.float)
        self.run_test(model, (x, ))

    def test_nn_sequential_inherit(self):
        class ConvBNReLU(nn.Sequential):
            def __init__(self):
                super().__init__(
                    nn.Conv2d(3, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(3),
                    nn.ReLU(inplace=False)
                )

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_bn_relu = ConvBNReLU()

            def forward(self, x):
                return self.conv_bn_relu(x)

        model = Net()
        x = torch.rand((1, 3, 224, 224), dtype=torch.float)
        self.run_test(model, (x, ))


class TestModelsWithShape(TestModels, ConvertWithShapeMixin):
    pass
