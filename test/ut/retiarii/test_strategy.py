import random
import time
import threading
from typing import *

import nni.retiarii.execution.api
import nni.retiarii.nn.pytorch as nn
import nni.retiarii.strategy as strategy
import torch
import torch.nn.functional as F
from nni.retiarii import Model
from nni.retiarii.converter import convert_to_graph
from nni.retiarii.execution import wait_models
from nni.retiarii.execution.interface import AbstractExecutionEngine, WorkerInfo, MetricData, AbstractGraphListener
from nni.retiarii.graph import DebugTraining, ModelStatus
from nni.retiarii.nn.pytorch.mutator import process_inline_mutation


class MockExecutionEngine(AbstractExecutionEngine):
    def __init__(self, failure_prob=0.):
        self.models = []
        self.failure_prob = failure_prob
        self._resource_left = 4

    def _model_complete(self, model: Model):
        time.sleep(random.uniform(0, 1))
        if random.uniform(0, 1) < self.failure_prob:
            model.status = ModelStatus.Failed
        else:
            model.metric = random.uniform(0, 1)
            model.status = ModelStatus.Trained
        self._resource_left += 1

    def submit_models(self, *models: Model) -> None:
        for model in models:
            self.models.append(model)
            self._resource_left -= 1
            threading.Thread(target=self._model_complete, args=(model, )).start()

    def query_available_resource(self) -> Union[List[WorkerInfo], int]:
        return self._resource_left

    def register_graph_listener(self, listener: AbstractGraphListener) -> None:
        pass

    def trial_execute_graph(cls) -> MetricData:
        pass


def _reset_execution_engine(engine=None):
    nni.retiarii.execution.api._execution_engine = engine


class Net(nn.Module):
    def __init__(self, hidden_size=32):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.LayerChoice([
            nn.Linear(4*4*50, hidden_size, bias=True),
            nn.Linear(4*4*50, hidden_size, bias=False)
        ])
        self.fc2 = nn.LayerChoice([
            nn.Linear(hidden_size, 10, bias=False),
            nn.Linear(hidden_size, 10, bias=True)
        ])

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def _get_model_and_mutators():
    base_model = Net()
    script_module = torch.jit.script(base_model)
    base_model_ir = convert_to_graph(script_module, base_model)
    base_model_ir.training_config = DebugTraining()
    mutators = process_inline_mutation(base_model_ir)
    return base_model_ir, mutators


def test_grid_search():
    gridsearch = strategy.GridSearch()
    engine = MockExecutionEngine()
    _reset_execution_engine(engine)
    gridsearch.run(*_get_model_and_mutators())
    wait_models(*engine.models)
    selection = set()
    for model in engine.models:
        selection.add((
            model.get_node_by_name('_model__fc1').operation.parameters['bias'],
            model.get_node_by_name('_model__fc2').operation.parameters['bias']
        ))
    assert len(selection) == 4
    _reset_execution_engine()


def test_random_search():
    random = strategy.Random()
    engine = MockExecutionEngine()
    _reset_execution_engine(engine)
    random.run(*_get_model_and_mutators())
    wait_models(*engine.models)
    selection = set()
    for model in engine.models:
        selection.add((
            model.get_node_by_name('_model__fc1').operation.parameters['bias'],
            model.get_node_by_name('_model__fc2').operation.parameters['bias']
        ))
    assert len(selection) == 4
    _reset_execution_engine()


def test_evolution():
    evolution = strategy.RegularizedEvolution(population_size=5, sample_size=3, cycles=10, mutation_prob=0.5, on_failure='ignore')
    engine = MockExecutionEngine(failure_prob=0.2)
    _reset_execution_engine(engine)
    evolution.run(*_get_model_and_mutators())
    wait_models(*engine.models)
    _reset_execution_engine()

    evolution = strategy.RegularizedEvolution(population_size=5, sample_size=3, cycles=10, mutation_prob=0.5, on_failure='worst')
    engine = MockExecutionEngine(failure_prob=0.4)
    _reset_execution_engine(engine)
    evolution.run(*_get_model_and_mutators())
    wait_models(*engine.models)
    _reset_execution_engine()


if __name__ == '__main__':
    test_grid_search()
    test_random_search()
    test_evolution()
