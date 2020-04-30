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
from unittest import TestCase, main

from nni.graph_utils import build_module_graph, build_graph

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

def read_expected_content(function_ptr):
    expected_file = os.path.join(os.path.dirname(__file__), "expect", "test_pytorch_graph.expect")
    assert os.path.exists(expected_file), expected_file
    with open(expected_file, "r") as f:
        return f.read()

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

    def test_pytorch_graph(self):
        dummy_input = (torch.zeros(1, 3),)

        class myLinear(torch.nn.Module):
            def __init__(self):
                super(myLinear, self).__init__()
                self.l = torch.nn.Linear(3, 5)

            def forward(self, x):
                return self.l(x)

        actual_proto, _ = build_graph(myLinear(), dummy_input)

        expected_str = read_expected_content(self)
        expected_proto = GraphDef()
        text_format.Parse(expected_str, expected_proto)

        self.assertEquals(len(expected_proto.node), len(actual_proto.node))
        for i in range(len(expected_proto.node)):
            expected_node = expected_proto.node[i]
            actual_node = actual_proto.node[i]
            self.assertEquals(expected_node.name, actual_node.name)
            self.assertEquals(expected_node.op, actual_node.op)
            self.assertEquals(expected_node.input, actual_node.input)
            self.assertEquals(expected_node.device, actual_node.device)
            self.assertEquals(
                sorted(expected_node.attr.keys()), sorted(actual_node.attr.keys()))

if __name__ == '__main__':
    main()
