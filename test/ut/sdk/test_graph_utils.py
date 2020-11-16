# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import os
import math
import uuid
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboard.compat.proto.graph_pb2 import GraphDef
from google.protobuf import text_format
import unittest
from unittest import TestCase, main

from nni.common.graph_utils import build_module_graph, build_graph, TorchModuleGraph, TUPLE_UNPACK_KIND

class BackboneModel1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 1, 1)
    def forward(self, x):
        return self.conv1(x)

class BackboneModel2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)
        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class BigModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone1 = BackboneModel1()
        self.backbone2 = BackboneModel2()
        self.fc3 = nn.Linear(10, 2) 
    def forward(self, x):
        x = self.backbone1(x)
        x = self.backbone2(x)
        x = self.fc3(x)
        return x

@unittest.skipIf(torch.__version__ >= '1.6.0', 'not supported')
class GraphUtilsTestCase(TestCase):
    def test_build_module_graph(self):
        big_model = BigModel()
        g = build_module_graph(big_model, torch.randn(2, 1, 28, 28))
        print(g.name_to_node.keys())
        leaf_modules = set([
            'backbone1.conv1', 'backbone2.bn1', 'backbone2.bn2', 'backbone2.conv1',
            'backbone2.conv2', 'backbone2.fc1', 'backbone2.fc2', 'fc3'
        ])

        assert set(g.leaf_modules) == leaf_modules
        assert not leaf_modules - set(g.name_to_node.keys())
        assert g.find_successors('backbone2.conv1') == ['backbone2.bn1']
        assert g.find_successors('backbone2.conv2') == ['backbone2.bn2']
        assert g.find_predecessors('backbone2.bn1') == ['backbone2.conv1']
        assert g.find_predecessors('backbone2.bn2') == ['backbone2.conv2']

    def _test_graph(self, model, dummy_input, expected_file):
        actual_proto, _ = build_graph(model, dummy_input)

        assert os.path.exists(expected_file), expected_file
        with open(expected_file, "r") as f:
            expected_str = f.read()

        expected_proto = GraphDef()
        text_format.Parse(expected_str, expected_proto)

        self.assertEqual(len(expected_proto.node), len(actual_proto.node))
        for i in range(len(expected_proto.node)):
            expected_node = expected_proto.node[i]
            actual_node = actual_proto.node[i]
            self.assertEqual(expected_node.name, actual_node.name)
            self.assertEqual(expected_node.op, actual_node.op)
            self.assertEqual(expected_node.input, actual_node.input)
            self.assertEqual(expected_node.device, actual_node.device)
            self.assertEqual(
                sorted(expected_node.attr.keys()), sorted(actual_node.attr.keys()))

    @unittest.skipIf(torch.__version__ < "1.4.0", "not supported")
    def test_graph_module1(self):
        dummy_input = (torch.zeros(1, 3),)

        class myLinear(torch.nn.Module):
            def __init__(self):
                super(myLinear, self).__init__()
                self.l = torch.nn.Linear(3, 5)

            def forward(self, x):
                return self.l(x)

        self._test_graph(
            myLinear(),
            dummy_input,
            os.path.join(os.path.dirname(__file__), "expect", "test_graph_module1.expect")
        )

    @unittest.skipIf(torch.__version__ < "1.4.0", "not supported")
    def test_graph_module2(self):
        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Linear(5, 3)
                self.bias = nn.Linear(5, 3)
                self.module = nn.Linear(6, 1)

            def forward(self, x):
                tensors = [self.weight(x), self.bias(x)]
                self.module(torch.cat(tensors, dim=1))
                return x

        self._test_graph(
            MyModule(),
            torch.randn(4, 5),
            os.path.join(os.path.dirname(__file__), "expect", "test_graph_module2.expect")
        )

    @unittest.skipIf(torch.__version__ < "1.4.0", "not supported")
    def test_graph_module3(self):
        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.module = nn.ModuleList([
                    nn.Linear(5, 3),
                    nn.Linear(3, 1)
                ])

            def forward(self, x):
                x = self.module[0](x)
                x = self.module[1](x)
                return x

        self._test_graph(
            MyModule(),
            torch.randn(4, 5),
            os.path.join(os.path.dirname(__file__), "expect", "test_graph_module3.expect")
        )
    
    @unittest.skipIf(torch.__version__ < "1.4.0", "not supported")
    def test_module_reuse(self):
        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.liner1 = nn.Linear(10, 10)
                self.relu = nn.ReLU(inplace=True)
                self.liner2 = nn.Linear(10, 20)
                self.liner3 = nn.Linear(20, 10)

            def forward(self, x):
                x = self.liner1(x)
                x = self.relu(x)
                x = self.liner2(x)
                x = self.relu(x)
                x = self.liner3(x)
                x = self.relu(x)
                return x

        data = torch.rand(10, 10)
        net = MyModule()
        traced = torch.jit.trace(net, data)
        modulegraph = TorchModuleGraph(traced_model=traced)
        # Traverse the TorchModuleGraph, due the resue of the relu module,
        # there will be three cpp_nodes corrspoding to the same module.
        # During traversing the graph, there should be only one
        # successor of each cpp-node (including the cpp_nodes that corresponds
        # to the same relu module).
        for name, nodeio in modulegraph.nodes_py.nodes_io.items():
            if nodeio.input_or_output == 'input':
                # Find the first node of the whole graph
                start_nodes = modulegraph.input_to_node[name]
                # We have only one single path top-down
                assert len(start_nodes) == 1
                node = start_nodes[0].unique_name
                while modulegraph.find_successors(node):
                    nodes = modulegraph.find_successors(node)
                    assert len(nodes) == 1
                    node = nodes[0]

    @unittest.skipIf(torch.__version__ < "1.4.0", "not supported")
    def test_module_unpack(self):
        """
        test the tuple/list unpack function of TorchModuleGraph.
        Following models are from the issue 2756
        https://github.com/microsoft/nni/issues/2756.
        MyModule will have two successive tuple unpack operations
        between the B and C.
        """
        class CBR(nn.Module):
            def __init__(self, i, o):
                super(CBR, self).__init__()
                self.conv1 = nn.Conv2d(i, o, kernel_size=1)
                self.bn1 = nn.BatchNorm2d(o)
                self.act1 = nn.ReLU()

            def forward(self, x):
                return self.act1(self.bn1(self.conv1(x)))


        class A(nn.Module):
            def __init__(self):
                super(A, self).__init__()
                self.conv1 = CBR(3, 6, )
                self.conv2 = CBR(6, 8, )
                self.conv3 = CBR(6, 12)

            def forward(self, x):
                x1 = self.conv1(x)
                x2 = self.conv2(x1)
                x3 = self.conv3(x1)
                return (x2, x3)


        class B1(nn.Module):
            def __init__(self):
                super(B1, self).__init__()
                self.conv1 = CBR(12, 32)
                self.conv2 = CBR(32, 32)
                self.conv3 = CBR(32, 32)

            def forward(self, x):
                x1 = self.conv1(x)
                x2 = self.conv2(x1)
                x3 = self.conv3(x2)
                return (x1, x2, x3)

        class B(nn.Module):
            def __init__(self):
                super(B, self).__init__()
                self.b = B1()

            def forward(self, x):
                return self.b(x[-1])

        class C(nn.Module):
            def __init__(self):
                super(C, self).__init__()
                self.conv1 = CBR(8, 32)
                self.conv2 = CBR(12, 32)
                self.conv3 = CBR(32, 32)
                self.conv4 = CBR(32, 32)
                self.conv5 = CBR(32, 32)

            def forward(self, x):
                return(self.conv1(x[0]), self.conv2(x[1]), self.conv3(x[2]),self.conv4(x[3]),self.conv5(x[4]))

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.a = A()
                self.b = B()
                # self.dummy = Dummy()
                self.c = C()

            def forward(self, x):
                x_a = self.a(x)
                x_b = self.b(x_a)
                xc = self.c(x_a + x_b)
                return xc

        dummy_input = torch.rand(1, 3, 28, 28)
        model = MyModule()
        graph = TorchModuleGraph(model, dummy_input)
        graph.unpack_manually()
        for node in graph.nodes_py.nodes_op:
            # The input of the function nodes should
            # not come from the TupleUnpack node, because
            # all the TupleUnpack nodes have been removed(unpacked)
            # manually
            for _input in node.inputs:
                if _input in graph.output_to_node:
                    preprocessor = graph.output_to_node[_input]
                    assert preprocessor.op_type != TUPLE_UNPACK_KIND


if __name__ == '__main__':
    main()
