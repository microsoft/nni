# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from logging import critical
import unittest

import torch
import torch.nn.functional as F

from nni.algorithms.compression.v2.pytorch.pruning import (
    LevelPruner,
    L1NormPruner,
    L2NormPruner,
    SlimPruner,
    FPGMPruner,
    ActivationAPoZRankPruner,
    ActivationMeanRankPruner,
    TaylorFOWeightPruner
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


def trainer(model, optimizer, criterion):
    model.train()
    input = torch.rand(10, 1, 28, 28)
    label = torch.Tensor(list(range(10))).type(torch.LongTensor)
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()


def get_optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)


criterion = torch.nn.CrossEntropyLoss()


class PrunerTestCase(unittest.TestCase):
    def test_level_pruner(self):
        model = TorchModel()
        config_list = [{'op_types': ['Conv2d'], 'sparsity': 0.8}]
        pruner = LevelPruner(model=model, config_list=config_list)
        pruned_model, masks = pruner.compress()
        # TODO: validate masks sparsity

    def test_l1_norm_pruner(self):
        model = TorchModel()
        config_list = [{'op_types': ['Conv2d'], 'sparsity': 0.8}]
        pruner = L1NormPruner(model=model, config_list=config_list, mode='dependency_aware',
                              dummy_input=torch.rand(10, 1, 28, 28))
        pruned_model, masks = pruner.compress()

    def test_l2_norm_pruner(self):
        model = TorchModel()
        config_list = [{'op_types': ['Conv2d'], 'sparsity': 0.8}]
        pruner = L2NormPruner(model=model, config_list=config_list, mode='dependency_aware',
                              dummy_input=torch.rand(10, 1, 28, 28))
        pruned_model, masks = pruner.compress()

    def test_fpgm_pruner(self):
        model = TorchModel()
        config_list = [{'op_types': ['Conv2d'], 'sparsity': 0.8}]
        pruner = FPGMPruner(model=model, config_list=config_list, mode='dependency_aware',
                            dummy_input=torch.rand(10, 1, 28, 28))
        pruned_model, masks = pruner.compress()

    def test_slim_pruner(self):
        model = TorchModel()
        config_list = [{'op_types': ['BatchNorm2d'], 'total_sparsity': 0.8}]
        pruner = SlimPruner(model=model, config_list=config_list, trainer=trainer, optimizer=get_optimizer(model),
                            criterion=criterion, training_epochs=1, scale=0.001, mode='global')
        pruned_model, masks = pruner.compress()

    def test_activation_apoz_rank_pruner(self):
        model = TorchModel()
        config_list = [{'op_types': ['Conv2d'], 'sparsity': 0.8}]
        pruner = ActivationAPoZRankPruner(model=model, config_list=config_list, trainer=trainer,
                                          optimizer=get_optimizer(model), criterion=criterion, training_batches=1,
                                          activation='relu', mode='dependency_aware',
                                          dummy_input=torch.rand(10, 1, 28, 28))
        pruned_model, masks = pruner.compress()

    def test_activation_mean_rank_pruner(self):
        model = TorchModel()
        config_list = [{'op_types': ['Conv2d'], 'sparsity': 0.8}]
        pruner = ActivationMeanRankPruner(model=model, config_list=config_list, trainer=trainer,
                                          optimizer=get_optimizer(model), criterion=criterion, training_batches=1,
                                          activation='relu', mode='dependency_aware',
                                          dummy_input=torch.rand(10, 1, 28, 28))
        pruned_model, masks = pruner.compress()

    def test_taylor_fo_pruner(self):
        model = TorchModel()
        config_list = [{'op_types': ['Conv2d'], 'sparsity': 0.8}]
        pruner = TaylorFOWeightPruner(model=model, config_list=config_list, trainer=trainer,
                                      optimizer=get_optimizer(model), criterion=criterion, training_batches=1,
                                      mode='dependency_aware', dummy_input=torch.rand(10, 1, 28, 28))
        pruned_model, masks = pruner.compress()


if __name__ == '__main__':
    unittest.main()
