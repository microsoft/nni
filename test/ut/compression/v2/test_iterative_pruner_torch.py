# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
import unittest

import torch
import torch.nn.functional as F

import nni
from nni.algorithms.compression.v2.pytorch.pruning import (
    LinearPruner,
    AGPPruner,
    LotteryTicketPruner,
    SimulatedAnnealingPruner,
    AutoCompressPruner,
    AMCPruner
)
from nni.algorithms.compression.v2.pytorch.utils import compute_sparsity_mask2compact


class TorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, 5, 1)
        self.bn1 = torch.nn.BatchNorm2d(10)
        self.conv2 = torch.nn.Conv2d(10, 10, 5, 1)
        self.bn2 = torch.nn.BatchNorm2d(10)
        self.fc1 = torch.nn.Linear(4 * 4 * 10, 100)
        self.fc2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def trainer(model, optimizer, criterion):
    model.train()
    for _ in range(10):
        input = torch.rand(10, 1, 28, 28)
        label = torch.Tensor(list(range(10))).type(torch.LongTensor)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()


def get_optimizer(model):
    return nni.trace(torch.optim.SGD)(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)


criterion = torch.nn.CrossEntropyLoss()


def evaluator(model):
    return random.random()


def finetuner(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    trainer(model, optimizer, criterion)


class IterativePrunerTestCase(unittest.TestCase):
    def test_linear_pruner(self):
        model = TorchModel()
        config_list = [{'op_types': ['Conv2d'], 'sparsity': 0.8}]
        pruner = LinearPruner(model, config_list, 'level', 3, log_dir='../../../logs')
        pruner.compress()
        _, pruned_model, masks, _, _ = pruner.get_best_result()
        sparsity_list = compute_sparsity_mask2compact(pruned_model, masks, config_list)
        assert 0.78 < sparsity_list[0]['total_sparsity'] < 0.82

    def test_agp_pruner(self):
        model = TorchModel()
        config_list = [{'op_types': ['Conv2d'], 'sparsity': 0.8}]
        pruner = AGPPruner(model, config_list, 'level', 3, log_dir='../../../logs')
        pruner.compress()
        _, pruned_model, masks, _, _ = pruner.get_best_result()
        sparsity_list = compute_sparsity_mask2compact(pruned_model, masks, config_list)
        assert 0.78 < sparsity_list[0]['total_sparsity'] < 0.82

    def test_lottery_ticket_pruner(self):
        model = TorchModel()
        config_list = [{'op_types': ['Conv2d'], 'sparsity': 0.8}]
        pruner = LotteryTicketPruner(model, config_list, 'level', 3, log_dir='../../../logs')
        pruner.compress()
        _, pruned_model, masks, _, _ = pruner.get_best_result()
        sparsity_list = compute_sparsity_mask2compact(pruned_model, masks, config_list)
        assert 0.78 < sparsity_list[0]['total_sparsity'] < 0.82

    def test_simulated_annealing_pruner(self):
        model = TorchModel()
        config_list = [{'op_types': ['Conv2d'], 'total_sparsity': 0.8}]
        pruner = SimulatedAnnealingPruner(model, config_list, evaluator, start_temperature=40, log_dir='../../../logs')
        pruner.compress()
        _, pruned_model, masks, _, _ = pruner.get_best_result()
        sparsity_list = compute_sparsity_mask2compact(pruned_model, masks, config_list)
        assert 0.78 < sparsity_list[0]['total_sparsity'] < 0.82

    def test_auto_compress_pruner(self):
        model = TorchModel()
        config_list = [{'op_types': ['Conv2d'], 'total_sparsity': 0.8}]
        admm_params = {
            'trainer': trainer,
            'traced_optimizer': get_optimizer(model),
            'criterion': criterion,
            'iterations': 10,
            'training_epochs': 1
        }
        sa_params = {
            'evaluator': evaluator,
            'start_temperature': 40
        }
        pruner = AutoCompressPruner(model, config_list, 10, admm_params, sa_params=sa_params, log_dir='../../../logs')
        pruner.compress()
        _, pruned_model, masks, _, _ = pruner.get_best_result()
        sparsity_list = compute_sparsity_mask2compact(pruned_model, masks, config_list)
        print(sparsity_list)
        assert 0.78 < sparsity_list[0]['total_sparsity'] < 0.82

    def test_amc_pruner(self):
        model = TorchModel()
        config_list = [{'op_types': ['Conv2d'], 'total_sparsity': 0.5, 'max_sparsity_per_layer': 0.8}]
        dummy_input = torch.rand(10, 1, 28, 28)
        ddpg_params = {'hidden1': 300, 'hidden2': 300, 'lr_c': 1e-3, 'lr_a': 1e-4, 'warmup': 5, 'discount': 1.,
                       'bsize': 64, 'rmsize': 100, 'window_length': 1, 'tau': 0.01, 'init_delta': 0.5, 'delta_decay': 0.99,
                       'max_episode_length': 1e9, 'epsilon': 50000}
        pruner = AMCPruner(10, model, config_list, dummy_input, evaluator, finetuner=finetuner, ddpg_params=ddpg_params, target='flops', log_dir='../../../logs')
        pruner.compress()

if __name__ == '__main__':
    unittest.main()
