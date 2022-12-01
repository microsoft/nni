import os
import threading
import unittest
import time
import torch
import torch.nn as nn
from pytorch_lightning.utilities.seed import seed_everything

from pathlib import Path

import nni
from nni.experiment.config import RemoteConfig, RemoteMachineConfig
from nni.runtime.tuner_command_channel import legacy as protocol
import json

try:
    from nni.common.device import GPUDevice
    from nni.retiarii.execution.cgo_engine import CGOExecutionEngine
    from nni.retiarii import Model
    from nni.retiarii.graph import Node

    from nni.retiarii import Model, submit_models
    from nni.retiarii.integration import RetiariiAdvisor
    from nni.retiarii.execution import set_execution_engine
    from nni.retiarii.execution.logical_optimizer.opt_dedup_input import DedupInputOptimizer
    from nni.retiarii.execution.logical_optimizer.logical_plan import LogicalPlan
    from nni.retiarii.utils import import_

    from nni.retiarii import serialize
    import nni.retiarii.evaluator.pytorch.lightning as pl
    from nni.retiarii.evaluator.pytorch.cgo.evaluator import MultiModelSupervisedLearningModule, _MultiModelSupervisedLearningModule
    import nni.retiarii.evaluator.pytorch.cgo.trainer as cgo_trainer

    import nni.retiarii.integration_api

    module_import_failed = False
except ImportError:
    module_import_failed = True

import pytest
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.datasets import load_diabetes

pytestmark = pytest.mark.skip(reason='Will be rewritten.')


class _model_cpu(nn.Module):
    def __init__(self):
        super().__init__()
        self.M_1_stem = M_1_stem()
        self.M_2_stem = M_2_stem()
        self.M_1_flatten = torch.nn.Flatten()
        self.M_2_flatten = torch.nn.Flatten()
        self.M_1_fc1 = torch.nn.Linear(out_features=256, in_features=1024)
        self.M_2_fc1 = torch.nn.Linear(out_features=256, in_features=1024)
        self.M_1_fc2 = torch.nn.Linear(out_features=10, in_features=256)
        self.M_2_fc2 = torch.nn.Linear(out_features=10, in_features=256)
        self.M_1_softmax = torch.nn.Softmax()
        self.M_2_softmax = torch.nn.Softmax()

    def forward(self, *_inputs):
        M_1__inputs_to_M_2_stem = _inputs[0]
        M_1_stem = self.M_1_stem(_inputs[0])
        M_2_stem = self.M_2_stem(M_1__inputs_to_M_2_stem)
        M_1_flatten = self.M_1_flatten(M_1_stem)
        M_2_flatten = self.M_2_flatten(M_2_stem)
        M_1_fc1 = self.M_1_fc1(M_1_flatten)
        M_2_fc1 = self.M_2_fc1(M_2_flatten)
        M_1_fc2 = self.M_1_fc2(M_1_fc1)
        M_2_fc2 = self.M_2_fc2(M_2_fc1)
        M_1_softmax = self.M_1_softmax(M_1_fc2)
        M_2_softmax = self.M_2_softmax(M_2_fc2)
        return M_1_softmax, M_2_softmax


class _model_gpu(nn.Module):
    def __init__(self):
        super().__init__()
        self.M_1_stem = M_1_stem().to('cuda:0')
        self.M_2_stem = M_2_stem().to('cuda:1')
        self.M_1_flatten = torch.nn.Flatten().to('cuda:0')
        self.M_2_flatten = torch.nn.Flatten().to('cuda:1')
        self.M_1_fc1 = torch.nn.Linear(out_features=256, in_features=1024).to('cuda:0')
        self.M_2_fc1 = torch.nn.Linear(out_features=256, in_features=1024).to('cuda:1')
        self.M_1_fc2 = torch.nn.Linear(out_features=10, in_features=256).to('cuda:0')
        self.M_2_fc2 = torch.nn.Linear(out_features=10, in_features=256).to('cuda:1')
        self.M_1_softmax = torch.nn.Softmax().to('cuda:0')
        self.M_2_softmax = torch.nn.Softmax().to('cuda:1')

    def forward(self, *_inputs):
        M_1__inputs_to_M_1_stem = _inputs[0].to("cuda:0")
        M_1__inputs_to_M_2_stem = _inputs[0].to("cuda:1")
        M_1_stem = self.M_1_stem(M_1__inputs_to_M_1_stem)
        M_2_stem = self.M_2_stem(M_1__inputs_to_M_2_stem)
        M_1_flatten = self.M_1_flatten(M_1_stem)
        M_2_flatten = self.M_2_flatten(M_2_stem)
        M_1_fc1 = self.M_1_fc1(M_1_flatten)
        M_2_fc1 = self.M_2_fc1(M_2_flatten)
        M_1_fc2 = self.M_1_fc2(M_1_fc1)
        M_2_fc2 = self.M_2_fc2(M_2_fc1)
        M_1_softmax = self.M_1_softmax(M_1_fc2)
        M_2_softmax = self.M_2_softmax(M_2_fc2)
        return M_1_softmax, M_2_softmax


class M_1_stem(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(out_channels=32, in_channels=1, kernel_size=5)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(out_channels=64, in_channels=32, kernel_size=5)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2)

    def forward(self, *_inputs):
        conv1 = self.conv1(_inputs[0])
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        return pool2


class M_2_stem(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(out_channels=32, in_channels=1, kernel_size=5)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(out_channels=64, in_channels=32, kernel_size=5)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2)

    def forward(self, *_inputs):
        conv1 = self.conv1(_inputs[0])
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        return pool2


def _reset():
    # this is to not affect other tests in sdk
    nni.trial._intermediate_seq = 0
    nni.trial._params = {'foo': 'bar', 'parameter_id': 0, 'parameters': {}}
    nni.runtime.platform.test._last_metric = None
    nni.retiarii.integration_api._advisor = None
    nni.retiarii.execution.api._execution_engine = None
    
    seed_everything(42)


def _new_trainer():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = serialize(MNIST, root='data/mnist', train=True, download=True, transform=transform)
    test_dataset = serialize(MNIST, root='data/mnist', train=False, download=True, transform=transform)

    multi_module = _MultiModelSupervisedLearningModule(nn.CrossEntropyLoss, {'acc': pl.AccuracyWithLogits})

    lightning = pl.Lightning(multi_module, cgo_trainer.Trainer(use_cgo=True,
                                                               max_epochs=1,
                                                               limit_train_batches=0.25,
                                                               enable_progress_bar=False),
                             train_dataloader=pl.DataLoader(train_dataset, batch_size=100),
                             val_dataloaders=pl.DataLoader(test_dataset, batch_size=100))
    return lightning


def _load_mnist(n_models: int = 1):
    path = Path('ut/nas/mnist_pytorch.json')
    with open(path) as f:
        mnist_model = Model._load(nni.load(fp=f))
        mnist_model.evaluator = _new_trainer()

    if n_models == 1:
        return mnist_model
    else:
        models = [mnist_model]
        for i in range(n_models - 1):
            forked_model = mnist_model.fork()
            forked_model.evaluator = _new_trainer()
            models.append(forked_model)
        return models


def _get_final_result():
    result = nni.load(nni.runtime.platform.test._last_metric)['value']
    if isinstance(result, list):
        return [float(_) for _ in result]
    else:
        if isinstance(result, str) and '[' in result:
            return nni.load(result)
        return [float(result)]


class CGOEngineTest(unittest.TestCase):
    def setUp(self):
        if module_import_failed:
            self.skipTest('test skip due to failed import of nni.retiarii.evaluator.pytorch.lightning')

    def test_multi_model_trainer_cpu(self):
        _reset()
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = serialize(MNIST, root='data/mnist', train=True, download=True, transform=transform)
        test_dataset = serialize(MNIST, root='data/mnist', train=False, download=True, transform=transform)

        multi_module = _MultiModelSupervisedLearningModule(nn.CrossEntropyLoss, {'acc': pl.AccuracyWithLogits}, n_models=2)

        lightning = pl.Lightning(multi_module, cgo_trainer.Trainer(use_cgo=True,
                                                                   max_epochs=1,
                                                                   limit_train_batches=0.25),
                                 train_dataloader=pl.DataLoader(train_dataset, batch_size=100),
                                 val_dataloaders=pl.DataLoader(test_dataset, batch_size=100))

        lightning._execute(_model_cpu)

        result = _get_final_result()
        assert len(result) == 2

        for _ in result:
            assert _ > 0.8

    def test_multi_model_trainer_gpu(self):
        _reset()
        if not (torch.cuda.is_available() and torch.cuda.device_count() >= 2):
            pytest.skip('test requires GPU and torch+cuda')
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = serialize(MNIST, root='data/mnist', train=True, download=True, transform=transform)
        test_dataset = serialize(MNIST, root='data/mnist', train=False, download=True, transform=transform)

        multi_module = _MultiModelSupervisedLearningModule(nn.CrossEntropyLoss, {'acc': pl.AccuracyWithLogits}, n_models=2)

        lightning = pl.Lightning(multi_module, cgo_trainer.Trainer(use_cgo=True,
                                                                   max_epochs=1,
                                                                   limit_train_batches=0.25),
                                 train_dataloader=pl.DataLoader(train_dataset, batch_size=100),
                                 val_dataloaders=pl.DataLoader(test_dataset, batch_size=100))

        lightning._execute(_model_gpu)

        result = _get_final_result()
        assert len(result) == 2

        for _ in result:
            assert _ > 0.8

    def _build_logical_with_mnist(self, n_models: int):
        lp = LogicalPlan()
        models = _load_mnist(n_models=n_models)
        for m in models:
            lp.add_model(m)
        return lp, models

    def test_add_model(self):
        _reset()

        lp, models = self._build_logical_with_mnist(3)

        for node in lp.logical_graph.hidden_nodes:
            old_nodes = [m.root_graph.get_node_by_id(node.id) for m in models]

            self.assertTrue(any([old_nodes[0].__repr__() == Node.__repr__(x) for x in old_nodes]))

    def test_dedup_input_four_devices(self):
        _reset()

        lp, models = self._build_logical_with_mnist(3)

        opt = DedupInputOptimizer()
        opt.convert(lp)

        advisor = RetiariiAdvisor('ws://_unittest_placeholder_')
        advisor._channel = protocol.LegacyCommandChannel()
        advisor.default_worker.start()
        advisor.assessor_worker.start()

        remote = RemoteConfig(machine_list=[])
        remote.machine_list.append(RemoteMachineConfig(host='test', gpu_indices=[0,1,2,3]))
        cgo = CGOExecutionEngine(training_service=remote, batch_waiting_time=0)

        phy_models = cgo._assemble(lp)
        self.assertTrue(len(phy_models) == 1)
        advisor.stopping = True
        advisor.default_worker.join()
        advisor.assessor_worker.join()
        cgo.join()

    def test_dedup_input_two_devices(self):
        _reset()

        lp, models = self._build_logical_with_mnist(3)

        opt = DedupInputOptimizer()
        opt.convert(lp)

        advisor = RetiariiAdvisor('ws://_unittest_placeholder_')
        advisor._channel = protocol.LegacyCommandChannel()
        advisor.default_worker.start()
        advisor.assessor_worker.start()

        remote = RemoteConfig(machine_list=[])
        remote.machine_list.append(RemoteMachineConfig(host='test', gpu_indices=[0,1]))
        cgo = CGOExecutionEngine(training_service=remote, batch_waiting_time=0)

        phy_models = cgo._assemble(lp)
        self.assertTrue(len(phy_models) == 2)
        advisor.stopping = True
        advisor.default_worker.join()
        advisor.assessor_worker.join()
        cgo.join()

    def test_submit_models(self):
        _reset()
        os.makedirs('generated', exist_ok=True)
        import nni.runtime.platform.test as tt
        protocol._set_out_file(open('generated/debug_protocol_out_file.py', 'wb'))
        protocol._set_in_file(open('generated/debug_protocol_out_file.py', 'rb'))

        models = _load_mnist(2)

        advisor = RetiariiAdvisor('ws://_unittest_placeholder_')
        advisor._channel = protocol.LegacyCommandChannel()
        advisor.default_worker.start()
        advisor.assessor_worker.start()
        # this is because RetiariiAdvisor only works after `_advisor_initialized` becomes True.
        # normally it becomes true when `handle_request_trial_jobs` is invoked
        advisor._advisor_initialized = True

        remote = RemoteConfig(machine_list=[])
        remote.machine_list.append(RemoteMachineConfig(host='test', gpu_indices=[0,1,2,3]))
        cgo_engine = CGOExecutionEngine(training_service=remote, batch_waiting_time=0)
        set_execution_engine(cgo_engine)
        submit_models(*models)
        time.sleep(3)

        if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            cmd, data = protocol.receive()
            params = nni.load(data)

            tt.init_params(params)

            trial_thread = threading.Thread(target=CGOExecutionEngine.trial_execute_graph)
            trial_thread.start()
            last_metric = None
            while True:
                time.sleep(1)
                if tt._last_metric:
                    metric = tt.get_last_metric()
                    if metric == last_metric:
                        continue
                    if 'value' in metric:
                        metric['value'] = json.dumps(metric['value'])
                    advisor.handle_report_metric_data(metric)
                    last_metric = metric
                if not trial_thread.is_alive():
                    trial_thread.join()
                    break

            trial_thread.join()

        advisor.stopping = True
        advisor.default_worker.join()
        advisor.assessor_worker.join()
        cgo_engine.join()


if __name__ == '__main__':
    unittest.main()
