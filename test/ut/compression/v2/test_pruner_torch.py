# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
import unittest

import numpy
import torch
import torch.nn.functional as F

import nni
from nni.compression.pytorch.pruning import (
    LevelPruner,
    L1NormPruner,
    L2NormPruner,
    SlimPruner,
    FPGMPruner,
    ActivationAPoZRankPruner,
    ActivationMeanRankPruner,
    TaylorFOWeightPruner,
    ADMMPruner,
    MovementPruner
)
from nni.algorithms.compression.v2.pytorch.utils import compute_sparsity_mask2compact


class TorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, 5, 1)
        self.bn1 = torch.nn.BatchNorm2d(10)
        self.conv2 = torch.nn.Conv2d(10, 20, 5, 1)
        self.bn2 = torch.nn.BatchNorm2d(20)
        self.fc1 = torch.nn.Linear(4 * 4 * 20, 100)
        self.fc2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 20)
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


class PrunerTestCase(unittest.TestCase):
    def test_level_pruner(self):
        model = TorchModel()
        config_list = [{'op_types': ['Conv2d'], 'sparsity': 0.8}]
        pruner = LevelPruner(model=model, config_list=config_list)
        pruned_model, masks = pruner.compress()
        pruner._unwrap_model()
        sparsity_list = compute_sparsity_mask2compact(pruned_model, masks, config_list)
        assert 0.78 < sparsity_list[0]['total_sparsity'] < 0.82

    def test_level_pruner_bank(self):
        model = TorchModel()
        config_list = [{'op_types': ['Conv2d'], 'sparsity': 0.7}]
        pruner = LevelPruner(model=model, config_list=config_list, mode='balance', balance_gran=[5])
        pruned_model, masks = pruner.compress()
        pruner._unwrap_model()
        sparsity_list = compute_sparsity_mask2compact(pruned_model, masks, config_list)
        # round down cause to lower sparsity
        assert sparsity_list[0]['total_sparsity'] == 0.6

    def test_l1_norm_pruner(self):
        model = TorchModel()
        config_list = [{'op_types': ['Conv2d'], 'sparsity': 0.8}]
        pruner = L1NormPruner(model=model, config_list=config_list, mode='dependency_aware',
                              dummy_input=torch.rand(10, 1, 28, 28))
        pruned_model, masks = pruner.compress()
        pruner._unwrap_model()
        sparsity_list = compute_sparsity_mask2compact(pruned_model, masks, config_list)
        assert 0.78 < sparsity_list[0]['total_sparsity'] < 0.82

    def test_l2_norm_pruner(self):
        model = TorchModel()
        config_list = [{'op_types': ['Conv2d'], 'sparsity': 0.8}]
        pruner = L2NormPruner(model=model, config_list=config_list, mode='dependency_aware',
                              dummy_input=torch.rand(10, 1, 28, 28))
        pruned_model, masks = pruner.compress()
        pruner._unwrap_model()
        sparsity_list = compute_sparsity_mask2compact(pruned_model, masks, config_list)
        assert 0.78 < sparsity_list[0]['total_sparsity'] < 0.82

    def test_fpgm_pruner(self):
        model = TorchModel()
        config_list = [{'op_types': ['Conv2d'], 'sparsity': 0.8}]
        pruner = FPGMPruner(model=model, config_list=config_list, mode='dependency_aware',
                            dummy_input=torch.rand(10, 1, 28, 28))
        pruned_model, masks = pruner.compress()
        pruner._unwrap_model()
        sparsity_list = compute_sparsity_mask2compact(pruned_model, masks, config_list)
        assert 0.78 < sparsity_list[0]['total_sparsity'] < 0.82

    def test_slim_pruner(self):
        model = TorchModel()
        config_list = [{'op_types': ['BatchNorm2d'], 'total_sparsity': 0.8}]
        pruner = SlimPruner(model=model, config_list=config_list, trainer=trainer, traced_optimizer=get_optimizer(model),
                            criterion=criterion, training_epochs=1, scale=0.001, mode='global')
        pruned_model, masks = pruner.compress()
        pruner._unwrap_model()
        sparsity_list = compute_sparsity_mask2compact(pruned_model, masks, config_list)
        assert 0.78 < sparsity_list[0]['total_sparsity'] < 0.82

    def test_activation_mean_rank_pruner(self):
        model = TorchModel()
        config_list = [{'op_types': ['Conv2d'], 'sparsity': 0.8}]
        pruner = ActivationMeanRankPruner(model=model, config_list=config_list, trainer=trainer,
                                          traced_optimizer=get_optimizer(model), criterion=criterion, training_batches=5,
                                          activation='relu', mode='dependency_aware',
                                          dummy_input=torch.rand(10, 1, 28, 28))
        pruned_model, masks = pruner.compress()
        pruner._unwrap_model()
        sparsity_list = compute_sparsity_mask2compact(pruned_model, masks, config_list)
        assert 0.78 < sparsity_list[0]['total_sparsity'] < 0.82

    def test_taylor_fo_pruner(self):
        model = TorchModel()
        config_list = [{'op_types': ['Conv2d'], 'sparsity': 0.8}]
        pruner = TaylorFOWeightPruner(model=model, config_list=config_list, trainer=trainer,
                                      traced_optimizer=get_optimizer(model), criterion=criterion, training_batches=5,
                                      mode='dependency_aware', dummy_input=torch.rand(10, 1, 28, 28))
        pruned_model, masks = pruner.compress()
        pruner._unwrap_model()
        sparsity_list = compute_sparsity_mask2compact(pruned_model, masks, config_list)
        assert 0.78 < sparsity_list[0]['total_sparsity'] < 0.82

    def test_admm_pruner(self):
        model = TorchModel()
        config_list = [{'op_types': ['Conv2d'], 'sparsity': 0.8, 'rho': 1e-3}]
        pruner = ADMMPruner(model=model, config_list=config_list, trainer=trainer, traced_optimizer=get_optimizer(model),
                            criterion=criterion, iterations=2, training_epochs=1)
        pruned_model, masks = pruner.compress()
        pruner._unwrap_model()
        sparsity_list = compute_sparsity_mask2compact(pruned_model, masks, config_list)
        assert 0.78 < sparsity_list[0]['total_sparsity'] < 0.82

    def test_movement_pruner(self):
        model = TorchModel()
        config_list = [{'op_types': ['Conv2d'], 'sparsity': 0.8}]
        pruner = MovementPruner(model=model, config_list=config_list, trainer=trainer, traced_optimizer=get_optimizer(model),
                                criterion=criterion, training_epochs=5, warm_up_step=0, cool_down_beginning_step=4)
        pruned_model, masks = pruner.compress()
        pruner._unwrap_model()
        sparsity_list = compute_sparsity_mask2compact(pruned_model, masks, config_list)
        assert 0.78 < sparsity_list[0]['total_sparsity'] < 0.82

class FixSeedPrunerTestCase(unittest.TestCase):
    def test_activation_apoz_rank_pruner(self):
        model = TorchModel()
        config_list = [{'op_types': ['Conv2d'], 'sparsity': 0.8}]
        pruner = ActivationAPoZRankPruner(model=model, config_list=config_list, trainer=trainer,
                                            traced_optimizer=get_optimizer(model), criterion=criterion, training_batches=5,
                                            activation='relu', mode='dependency_aware',
                                            dummy_input=torch.rand(10, 1, 28, 28))
        pruned_model, masks = pruner.compress()
        pruner._unwrap_model()
        sparsity_list = compute_sparsity_mask2compact(pruned_model, masks, config_list)
        assert 0.78 < sparsity_list[0]['total_sparsity'] < 0.82

    def setUp(self) -> None:
        # fix seed in order to solve the random failure of ut
        random.seed(1024)
        numpy.random.seed(1024)
        torch.manual_seed(1024)

    def tearDown(self) -> None:
        # reset seed
        import time
        now = int(time.time() * 100)
        random.seed(now)
        seed = random.randint(0, 2 ** 32 - 1)
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)

if __name__ == '__main__':
    unittest.main()
