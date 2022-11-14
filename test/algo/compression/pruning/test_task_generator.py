# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List
import unittest

import torch
import torch.nn.functional as F

from nni.algorithms.compression.v2.pytorch.base import TaskResult
from nni.algorithms.compression.v2.pytorch.pruning.tools import (
    AGPTaskGenerator,
    LinearTaskGenerator,
    LotteryTicketTaskGenerator,
    SimulatedAnnealingTaskGenerator
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


def run_task_generator(task_generator_type):
    model = TorchModel()
    config_list = [{'op_types': ['Conv2d'], 'sparsity': 0.8}]

    if task_generator_type == 'agp':
        task_generator = AGPTaskGenerator(5, model, config_list)
    elif task_generator_type == 'linear':
        task_generator = LinearTaskGenerator(5, model, config_list)
    elif task_generator_type == 'lottery_ticket':
        task_generator = LotteryTicketTaskGenerator(5, model, config_list)
    elif task_generator_type == 'simulated_annealing':
        task_generator = SimulatedAnnealingTaskGenerator(model, config_list)

    count = run_task_generator_(task_generator)

    if task_generator_type == 'agp':
        assert count == 6
    elif task_generator_type == 'linear':
        assert count == 6
    elif task_generator_type == 'lottery_ticket':
        assert count == 5
    elif task_generator_type == 'simulated_annealing':
        assert count == 17


def run_task_generator_(task_generator):
    task = task_generator.next()
    factor = 0.9
    count = 0
    while task is not None:
        factor = factor ** 2
        count += 1
        task_result = TaskResult(task.task_id, TorchModel(), {}, {}, 1 - factor)
        task_generator.receive_task_result(task_result)
        task = task_generator.next()
    return count


class TaskGenerator(unittest.TestCase):
    def test_agp_task_generator(self):
        run_task_generator('agp')

    def test_linear_task_generator(self):
        run_task_generator('linear')

    def test_lottery_ticket_task_generator(self):
        run_task_generator('lottery_ticket')

    def test_simulated_annealing_task_generator(self):
        run_task_generator('simulated_annealing')


if __name__ == '__main__':
    unittest.main()
