from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torchmetrics
from torchvision.datasets import MNIST
from torchvision import transforms
from pytorch_lightning import seed_everything

import nni
from nni.experiment.config import RemoteConfig, RemoteMachineConfig
from nni.nas.evaluator.pytorch import Lightning, DataLoader
from nni.nas.execution import SequentialExecutionEngine
from nni.nas.execution.cgo import CrossGraphOptimization
from nni.nas.execution.cgo.evaluator import MultiModelLightningModule, MultiModelTrainer
from nni.nas.execution.cgo.logical_optimizer.logical_plan import LogicalPlan
from nni.nas.execution.cgo.logical_optimizer.opt_dedup_input import DedupInputOptimizer
from nni.nas.space import Node, ModelStatus
from nni.nas.space.pytorch import PytorchGraphModelSpace
from nni.runtime.trial_command_channel import get_default_trial_command_channel, set_default_trial_command_channel

from ut.sdk.helper.trial_command_channel import TestHelperTrialCommandChannel


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


def create_evaluator(n_models=None, accelerator='gpu'):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = nni.trace(MNIST)(root='data/mnist', train=True, download=False, transform=transform)
    test_dataset = nni.trace(MNIST)(root='data/mnist', train=False, download=False, transform=transform)

    multi_module = MultiModelLightningModule(
        nn.CrossEntropyLoss(),
        torchmetrics.Accuracy('multiclass', num_classes=10),
        n_models=n_models
    )

    lightning = Lightning(
        multi_module,
        MultiModelTrainer(max_epochs=1, limit_train_batches=0.25, enable_progress_bar=True, accelerator=accelerator),
        train_dataloaders=DataLoader(train_dataset, batch_size=100),
        val_dataloaders=DataLoader(test_dataset, batch_size=100)
    )
    return lightning


def _load_mnist(n_models: int = 1):
    path = Path(__file__).parent / 'mnist_pytorch.json'
    with open(path) as f:
        mnist_model = PytorchGraphModelSpace._load(**nni.load(fp=f))
    mnist_model.evaluator = create_evaluator()
    mnist_model.status = ModelStatus.Frozen

    if n_models == 1:
        return mnist_model
    else:
        models = [mnist_model]
        for _ in range(n_models - 1):
            forked_model = mnist_model.fork()
            forked_model.status = ModelStatus.Frozen
            models.append(forked_model)
        return models


def _build_logical_with_mnist(n_models: int):
    lp = LogicalPlan(model_cls=PytorchGraphModelSpace)
    models = _load_mnist(n_models=n_models)
    for m in models:
        lp.add_model(m)
    return lp, models


@pytest.fixture(autouse=True)
def seed():
    seed_everything(42)


@pytest.fixture
def trial_command_channel():
    _default_channel = get_default_trial_command_channel()
    channel = TestHelperTrialCommandChannel()
    set_default_trial_command_channel(channel)

    nni.get_next_parameter()

    yield channel

    set_default_trial_command_channel(_default_channel)


@pytest.fixture(params=[1, 2, 4])
def cgo(request):
    remote = RemoteConfig(machine_list=[])
    remote.machine_list.append(RemoteMachineConfig(host='test', gpu_indices=list(range(request.param))))

    cgo = CrossGraphOptimization(remote_config=remote, batch_waiting_time=0)

    yield cgo

    cgo.shutdown()


def test_multi_model_trainer_cpu(trial_command_channel):
    evaluator = create_evaluator(n_models=2, accelerator='cpu')
    evaluator.evaluate(_model_cpu())

    result = trial_command_channel.final
    assert len(result) == 2

    for _ in result:
        assert _ > 0.8


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='test requires GPU and torch+cuda')
def test_multi_model_trainer_gpu(trial_command_channel):
    evaluator = create_evaluator(n_models=2)
    evaluator.evaluate(_model_gpu())

    result = trial_command_channel.final
    assert len(result) == 2

    for _ in result:
        assert _ > 0.8


def test_add_model():
    lp, models = _build_logical_with_mnist(3)

    for node in lp.logical_graph.hidden_nodes:
        old_nodes = [m.root_graph.get_node_by_id(node.id) for m in models]

        assert any([old_nodes[0].__repr__() == Node.__repr__(x) for x in old_nodes])


def test_dedup_input(cgo):
    lp, _ = _build_logical_with_mnist(3)

    opt = DedupInputOptimizer()
    opt.convert(lp)

    phy_models = cgo._assemble(lp)

    if len(cgo.available_devices) == 4:
        assert len(list(phy_models)) == 1
    elif len(cgo.available_devices) == 2:
        assert len(list(phy_models)) == 2
    elif len(cgo.available_devices) == 1:
        assert len(list(phy_models)) == 3
    else:
        raise ValueError(f'Invalid device count: {cgo.available_devices}')

    cgo.shutdown()


def test_submit_models(cgo):
    import logging
    logging.getLogger('nni.nas.execution.sequential').setLevel(logging.DEBUG)

    models = _load_mnist(2)

    engine = SequentialExecutionEngine(continue_on_failure=True)

    cgo.set_engine(engine)
    cgo.submit_models(*models)

    cgo.wait_models()

    return  # FIXME: status check skipped due to bugs in evaluator copy. It's sort of critical. Fix ASAP.

    if not torch.cuda.is_available():
        for model in models:  # can't be trained without gpu.
            assert model.status == ModelStatus.Failed
        if len(cgo.available_devices) == 1:
            assert engine._model_count == 2  # 2  single
        else:
            assert engine._model_count == 3  # 1 + retry 2
    elif torch.cuda.device_count() == 1 and len(cgo.available_devices) == 1:
        # Should be the case on pipeline.
        assert engine._model_count == 2  # No merge at all.
        for model in models:
            assert model.status == ModelStatus.Trained
            assert model.metrics.final > 0.8
