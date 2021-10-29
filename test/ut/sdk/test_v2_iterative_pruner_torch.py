# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
import unittest

import torch
import torch.nn.functional as F

from nni.algorithms.compression.v2.pytorch.pruning import (
    LinearPruner,
    AGPPruner,
    LotteryTicketPruner,
    SimulatedAnnealingPruner
)

from nni.algorithms.compression.v2.pytorch.utils import compute_sparsity_mask2compact


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


def evaluator(model):
    return random.random()


class IterativePrunerTestCase(unittest.TestCase):
    def test_linear_pruner(self):
        model = TorchModel()
        config_list = [{'op_types': ['Conv2d'], 'sparsity': 0.8}]
        pruner = LinearPruner(model, config_list, 'level', 3, log_dir='../../logs')
        pruner.compress()
        _, pruned_model, masks, _, _ = pruner.get_best_result()
        sparsity_list = compute_sparsity_mask2compact(pruned_model, masks, config_list)
        assert 0.79 < sparsity_list[0]['total_sparsity'] < 0.81

    def test_agp_pruner(self):
        model = TorchModel()
        config_list = [{'op_types': ['Conv2d'], 'sparsity': 0.8}]
        pruner = AGPPruner(model, config_list, 'level', 3, log_dir='../../logs')
        pruner.compress()
        _, pruned_model, masks, _, _ = pruner.get_best_result()
        sparsity_list = compute_sparsity_mask2compact(pruned_model, masks, config_list)
        assert 0.79 < sparsity_list[0]['total_sparsity'] < 0.81

    def test_lottery_ticket_pruner(self):
        model = TorchModel()
        config_list = [{'op_types': ['Conv2d'], 'sparsity': 0.8}]
        pruner = LotteryTicketPruner(model, config_list, 'level', 3, log_dir='../../logs')
        pruner.compress()
        _, pruned_model, masks, _, _ = pruner.get_best_result()
        sparsity_list = compute_sparsity_mask2compact(pruned_model, masks, config_list)
        assert 0.79 < sparsity_list[0]['total_sparsity'] < 0.81

    def test_simulated_annealing_pruner(self):
        model = TorchModel()
        config_list = [{'op_types': ['Conv2d'], 'sparsity': 0.8}]
        pruner = SimulatedAnnealingPruner(model, config_list, 'level', evaluator, start_temperature=30, log_dir='../../logs')
        pruner.compress()
        _, pruned_model, masks, _, _ = pruner.get_best_result()
        sparsity_list = compute_sparsity_mask2compact(pruned_model, masks, config_list)
        assert 0.79 < sparsity_list[0]['total_sparsity'] < 0.81

if __name__ == '__main__':
    unittest.main()
