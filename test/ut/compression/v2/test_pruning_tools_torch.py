# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import torch
import torch.nn.functional as F

import nni
from nni.algorithms.compression.v2.pytorch.base import Pruner
from nni.algorithms.compression.v2.pytorch.pruning.tools import (
    WeightDataCollector,
    WeightTrainerBasedDataCollector,
    SingleHookTrainerBasedDataCollector
)
from nni.algorithms.compression.v2.pytorch.pruning.tools import (
    NormMetricsCalculator,
    MultiDataNormMetricsCalculator,
    DistMetricsCalculator,
    APoZRankMetricsCalculator,
    MeanRankMetricsCalculator
)
from nni.algorithms.compression.v2.pytorch.pruning.tools import (
    NormalSparsityAllocator,
    GlobalSparsityAllocator
)
from nni.algorithms.compression.v2.pytorch.pruning.tools.base import HookCollectorInfo
from nni.algorithms.compression.v2.pytorch.utils import get_module_by_name
from nni.algorithms.compression.v2.pytorch.utils.constructor_helper import OptimizerConstructHelper


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


class PruningToolsTestCase(unittest.TestCase):
    def test_data_collector(self):
        model = TorchModel()
        w1 = torch.rand(5, 1, 5, 5)
        w2 = torch.rand(10, 5, 5, 5)
        model.conv1.weight.data = w1
        model.conv2.weight.data = w2

        config_list = [{'op_types': ['Conv2d']}]
        pruner = Pruner(model, config_list)

        # Test WeightDataCollector
        data_collector = WeightDataCollector(pruner)
        data = data_collector.collect()
        assert all(torch.equal(get_module_by_name(model, module_name)[1].module.weight.data, data[module_name]) for module_name in ['conv1', 'conv2'])

        # Test WeightTrainerBasedDataCollector
        def opt_after():
            model.conv1.module.weight.data = torch.ones(5, 1, 5, 5)
            model.conv2.module.weight.data = torch.ones(10, 5, 5, 5)

        optimizer_helper = OptimizerConstructHelper.from_trace(model, get_optimizer(model))
        data_collector = WeightTrainerBasedDataCollector(pruner, trainer, optimizer_helper, criterion, 1, opt_after_tasks=[opt_after])
        data = data_collector.collect()
        assert all(torch.equal(get_module_by_name(model, module_name)[1].module.weight.data, data[module_name]) for module_name in ['conv1', 'conv2'])
        assert all(t.numel() == (t == 1).type_as(t).sum().item() for t in data.values())

        # Test SingleHookTrainerBasedDataCollector
        def _collector(buffer, weight_tensor):
            def collect_taylor(grad):
                if len(buffer) < 2:
                    buffer.append(grad.clone().detach())
            return collect_taylor
        hook_targets = {'conv1': model.conv1.module.weight, 'conv2': model.conv2.module.weight}
        collector_info = HookCollectorInfo(hook_targets, 'tensor', _collector)

        optimizer_helper = OptimizerConstructHelper.from_trace(model, get_optimizer(model))
        data_collector = SingleHookTrainerBasedDataCollector(pruner, trainer, optimizer_helper, criterion, 2, collector_infos=[collector_info])
        data = data_collector.collect()
        assert all(len(t) == 2 for t in data.values())

    def test_metrics_calculator(self):
        # Test NormMetricsCalculator
        metrics_calculator = NormMetricsCalculator(dim=0, p=2)
        data = {
            '1': torch.ones(3, 3, 3),
            '2': torch.ones(4, 4) * 2
        }
        result = {
            '1': torch.ones(3) * 3,
            '2': torch.ones(4) * 4
        }
        metrics = metrics_calculator.calculate_metrics(data)
        assert all(torch.equal(result[k], v) for k, v in metrics.items())

        # Test DistMetricsCalculator
        metrics_calculator = DistMetricsCalculator(dim=0, p=2)
        data = {
            '1': torch.tensor([[1, 2], [4, 6]], dtype=torch.float32),
            '2': torch.tensor([[0, 0], [1, 1]], dtype=torch.float32)
        }
        result = {
            '1': torch.tensor([5, 5], dtype=torch.float32),
            '2': torch.sqrt(torch.tensor([2, 2], dtype=torch.float32))
        }
        metrics = metrics_calculator.calculate_metrics(data)
        assert all(torch.equal(result[k], v) for k, v in metrics.items())

        # Test MultiDataNormMetricsCalculator
        metrics_calculator = MultiDataNormMetricsCalculator(dim=0, p=1)
        data = {
            '1': [2, torch.ones(3, 3, 3) * 2],
            '2': [2, torch.ones(4, 4) * 2]
        }
        result = {
            '1': torch.ones(3) * 18,
            '2': torch.ones(4) * 8
        }
        metrics = metrics_calculator.calculate_metrics(data)
        assert all(torch.equal(result[k], v) for k, v in metrics.items())

        # Test APoZRankMetricsCalculator
        metrics_calculator = APoZRankMetricsCalculator(dim=1)
        data = {
            '1': [2, torch.tensor([[1, 1], [1, 1]], dtype=torch.float32)],
            '2': [2, torch.tensor([[0, 0, 1], [0, 0, 0]], dtype=torch.float32)]
        }
        result = {
            '1': torch.tensor([0.5, 0.5], dtype=torch.float32),
            '2': torch.tensor([1, 1, 0.75], dtype=torch.float32)
        }
        metrics = metrics_calculator.calculate_metrics(data)
        assert all(torch.equal(result[k], v) for k, v in metrics.items())

        # Test MeanRankMetricsCalculator
        metrics_calculator = MeanRankMetricsCalculator(dim=1)
        data = {
            '1': [2, torch.tensor([[0, 1], [1, 0]], dtype=torch.float32)],
            '2': [2, torch.tensor([[0, 0, 1], [0, 0, 0]], dtype=torch.float32)]
        }
        result = {
            '1': torch.tensor([0.25, 0.25], dtype=torch.float32),
            '2': torch.tensor([0, 0, 0.25], dtype=torch.float32)
        }
        metrics = metrics_calculator.calculate_metrics(data)
        assert all(torch.equal(result[k], v) for k, v in metrics.items())

    def test_sparsity_allocator(self):
        # Test NormalSparsityAllocator
        model = TorchModel()
        config_list = [{'op_types': ['Conv2d'], 'total_sparsity': 0.8}]
        pruner = Pruner(model, config_list)
        metrics = {
            'conv1': torch.rand(5, 1, 5, 5),
            'conv2': torch.rand(10, 5, 5, 5)
        }
        sparsity_allocator = NormalSparsityAllocator(pruner)
        masks = sparsity_allocator.generate_sparsity(metrics)
        assert all(v['weight'].sum() / v['weight'].numel() == 0.2 for k, v in masks.items())

        # Test GlobalSparsityAllocator
        model = TorchModel()
        config_list = [{'op_types': ['Conv2d'], 'total_sparsity': 0.8}]
        pruner = Pruner(model, config_list)
        sparsity_allocator = GlobalSparsityAllocator(pruner)
        masks = sparsity_allocator.generate_sparsity(metrics)
        total_elements, total_masked_elements = 0, 0
        for t in masks.values():
            total_elements += t['weight'].numel()
            total_masked_elements += t['weight'].sum().item()
        assert total_masked_elements / total_elements == 0.2


if __name__ == '__main__':
    unittest.main()
