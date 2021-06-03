import json
from nni.retiarii.graph import Node
import os
import threading
import unittest
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pathlib import Path

import nni
from nni.retiarii.execution.cgo_engine import CGOExecutionEngine
from nni.retiarii import Model

from nni.retiarii import Model, submit_models
from nni.retiarii.integration import RetiariiAdvisor
from nni.retiarii.execution import set_execution_engine
from nni.retiarii.evaluator.pytorch import PyTorchImageClassificationTrainer, PyTorchMultiModelTrainer
from nni.retiarii.execution.logical_optimizer.opt_dedup_input import DedupInputOptimizer
from nni.retiarii.execution.logical_optimizer.logical_plan import LogicalPlan
from nni.retiarii.utils import import_

from nni.retiarii import serialize_cls, serialize
import nni.retiarii.evaluator.pytorch.lightning as pl
from sklearn.datasets import load_diabetes
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST

from nni.retiarii.evaluator.pytorch.cgo_evaluator import BypassAccelerator, MultiModelSupervisedLearningModule

import pytest


debug = False

progress_bar_refresh_rate = 0
if debug:
    progress_bar_refresh_rate = 1


class _model_cpu(nn.Module):
    def __init__(self):
        super().__init__()
        self.M_1_stem = M_1_stem()  # .to('cuda:0')
        self.M_2_stem = M_2_stem()  # .to('cuda:1')
        self.M_1_flatten = torch.nn.Flatten()  # .to('cuda:0')
        self.M_2_flatten = torch.nn.Flatten()  # .to('cuda:1')
        self.M_1_fc1 = torch.nn.Linear(out_features=256, in_features=1024)  # .to('cuda:0')
        self.M_2_fc1 = torch.nn.Linear(out_features=256, in_features=1024)  # .to('cuda:1')
        self.M_1_fc2 = torch.nn.Linear(out_features=10, in_features=256)  # .to('cuda:0')
        self.M_2_fc2 = torch.nn.Linear(out_features=10, in_features=256)  # .to('cuda:1')
        self.M_1_softmax = torch.nn.Softmax()  # .to('cuda:0')
        self.M_2_softmax = torch.nn.Softmax()  # .to('cuda:1')

    def forward(self, *_inputs):
        M_1__inputs_to_M_2_stem = _inputs[0]  # .to("cuda:1")
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
        M_1__inputs_to_M_2_stem = _inputs[0].to("cuda:1")
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
        # self.to('cuda:1')
        # print(_inputs[0].get_device())
        # print(self.conv1.weight.get_device())
        conv1 = self.conv1(_inputs[0])
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        return pool2


def _reset():
    # this is to not affect other tests in sdk
    nni.trial._intermediate_seq = 0
    nni.trial._params = {'foo': 'bar', 'parameter_id': 0}
    nni.runtime.platform.test._last_metric = None
    nni.retiarii.integration_api._advisor = None
    nni.retiarii.execution.api._execution_engine = None


def _new_trainer():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = serialize(MNIST, root='data/mnist', train=True, download=True, transform=transform)
    test_dataset = serialize(MNIST, root='data/mnist', train=False, download=True, transform=transform)
    
    multi_module = MultiModelSupervisedLearningModule(nn.CrossEntropyLoss, {'acc': pl._AccuracyWithLogits}, n_models=100)

    lightning = pl.Lightning(multi_module, pl.Trainer(max_epochs=1,
                                                      limit_train_batches=0.25,
                                                      accelerator=BypassAccelerator(device='cuda:0')),
                             train_dataloader=pl.DataLoader(train_dataset, batch_size=100),
                             val_dataloaders=pl.DataLoader(test_dataset, batch_size=100))
    return lightning


def _load_mnist(n_models: int = 1):
    path = Path(__file__).parent / 'mnist_pytorch.json'
    with open(path) as f:
        mnist_model = Model._load(json.load(f))
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
    result = json.loads(nni.runtime.platform.test._last_metric)['value']
    if isinstance(result, list):
        return [float(_) for _ in result]
    else:
        if isinstance(result, str) and '[' in result:
            return json.loads(result)
        return [float(result)]


class CGOEngineTest(unittest.TestCase):

    def test_multi_model_trainer_cpu(self):
        _reset()
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = serialize(MNIST, root='data/mnist', train=True, download=True, transform=transform)
        test_dataset = serialize(MNIST, root='data/mnist', train=False, download=True, transform=transform)

        multi_module = MultiModelSupervisedLearningModule(nn.CrossEntropyLoss, {'acc': pl._AccuracyWithLogits}, n_models=2)

        lightning = pl.Lightning(multi_module, pl.Trainer(max_epochs=1,
                                                          limit_train_batches=0.25,
                                                          accelerator=BypassAccelerator(device='cpu')),
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

        multi_module = MultiModelSupervisedLearningModule(nn.CrossEntropyLoss, {'acc': pl._AccuracyWithLogits}, n_models=2)

        lightning = pl.Lightning(multi_module, pl.Trainer(max_epochs=1,
                                                          limit_train_batches=0.25,
                                                          accelerator=BypassAccelerator(device='cuda:0')),
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
        # TODO: topo_sort may not be stable that leads to different dump. skip
        # correct_json_path = Path(__file__).parent / 'dedup_logical_graph.json'
        # with open(correct_json_path , 'r') as fp:
        #     correct_dump = fp.readlines()
        #lp_dump = lp.logical_graph._dump()

        # self.assertTrue(correct_dump[0] == json.dumps(lp_dump))
        
        advisor = RetiariiAdvisor()
        available_devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
        cgo = CGOExecutionEngine(available_devices = available_devices)

        phy_models = cgo._assemble(lp)
        self.assertTrue(len(phy_models) == 1)
        advisor.stopping = True
        advisor.default_worker.join()
        advisor.assessor_worker.join()
    
    def test_dedup_input_two_devices(self):
        _reset()
        lp, models = self._build_logical_with_mnist(3)
        opt = DedupInputOptimizer()
        opt.convert(lp)
        # TODO: topo_sort may not be stable that leads to different dump. skip
        # correct_json_path = Path(__file__).parent / 'dedup_logical_graph.json'
        # with open(correct_json_path , 'r') as fp:
        #     correct_dump = fp.readlines()
        #lp_dump = lp.logical_graph._dump()

        # self.assertTrue(correct_dump[0] == json.dumps(lp_dump))
        
        advisor = RetiariiAdvisor()
        available_devices = ['cuda:0', 'cuda:1']
        cgo = CGOExecutionEngine(available_devices = available_devices)

        phy_models = cgo._assemble(lp)
        self.assertTrue(len(phy_models) == 2)
        advisor.stopping = True
        advisor.default_worker.join()
        advisor.assessor_worker.join()

    def test_submit_models(self):
        _reset()
        # os.environ['CGO_DEVICES'] = 'cuda:0,cuda:1,cuda:2,cuda:3'
        nni.retiarii.debug_configs.framework = 'pytorch'
        os.makedirs('generated', exist_ok=True)
        from nni.runtime import protocol
        import nni.runtime.platform.test as tt
        protocol._out_file = open('generated/debug_protocol_out_file.py', 'wb')
        protocol._in_file = open('generated/debug_protocol_out_file.py', 'rb')

        models = _load_mnist(2)
        advisor = RetiariiAdvisor()
        set_execution_engine(CGOExecutionEngine(available_devices=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']))
        submit_models(*models)

        if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            cmd, data = protocol.receive()
            params = json.loads(data)
            # params['parameters']['training_kwargs']['max_steps'] = 100

            tt.init_params(params)

            trial_thread = threading.Thread(target=CGOExecutionEngine.trial_execute_graph())
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
                    break

            trial_thread.join()
        advisor.stopping = True
        advisor.default_worker.join()
        advisor.assessor_worker.join()


if __name__ == '__main__':
    CGOEngineTest().test_dedup_input_two_devices()
    # unittest.main()
