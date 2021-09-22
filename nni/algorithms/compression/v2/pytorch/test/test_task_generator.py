# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import torch
import torch.nn.functional as F

from nni.algorithms.compression.v2.pytorch.base import Task, TaskResult
from nni.algorithms.compression.v2.pytorch.pruning import (
    AGPTaskGenerator,
    LinearTaskGenerator
)


class TorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 5, 5, 1)
        self.bn1 = torch.nn.BatchNorm2d(5)
        self.conv2 = torch.nn.Conv2d(5, 10, 5, 1)
        self.bn2 = torch.nn.BatchNorm2d(10)
        self.fc1 = torch.nn.Linear(4 * 4 * 10, 100)
        self.fc2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 10)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class TaskGenerator(unittest.TestCase):
    def test_agp_task_generator(self):
        model = TorchModel()
        config_list = [{'op_types': ['Conv2d'], 'sparsity': 0.8}]

        task_generator = AGPTaskGenerator(1, model, config_list)
        tasks = task_generator.generate_tasks()

        # TODO: wait for pr 4178
        pass

    def test_linear_task_generator(self):
        model = TorchModel()
        config_list = [{'op_types': ['Conv2d'], 'sparsity': 0.8}]

        task_generator = LinearTaskGenerator(1, model, config_list)
        tasks = task_generator.generate_tasks()

        # TODO: wait for pr 4178
        pass


if __name__ == '__main__':
    unittest.main()
