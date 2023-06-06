import pytest

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from torchvision import transforms
from torchvision.datasets import MNIST
from torch import nn
from torch.utils.data import Dataset, RandomSampler, Subset

import nni
from nni.nas import strategy
from nni.nas.execution import SequentialExecutionEngine
from nni.nas.space import RawFormatModelSpace, model_context
from nni.nas.evaluator.pytorch.lightning import Classification, Regression, ClassificationModule, DataLoader
from nni.nas.nn.pytorch import LayerChoice, ModelSpace
from nni.nas.oneshot.pytorch import DartsLightningModule

from ut.nas.nn.models import MODELS
from .test_utils import RandomDataset


pytestmark = pytest.mark.skipif(pl.__version__ < '1.0', reason='Incompatible APIs')


def _mnist_evaluator(**kwargs):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = nni.trace(MNIST)('data/mnist', download=False, train=True, transform=transform)
    # Multi-GPU combined dataloader will break this subset sampler. Expected though.
    train_random_sampler = nni.trace(RandomSampler)(train_dataset, True, int(len(train_dataset) / 20))
    train_loader = nni.trace(DataLoader)(train_dataset, 16, sampler=train_random_sampler)
    valid_dataset = nni.trace(MNIST)('data/mnist', download=False, train=False, transform=transform)
    valid_random_sampler = nni.trace(RandomSampler)(valid_dataset, True, int(len(valid_dataset) / 20))
    valid_loader = nni.trace(DataLoader)(valid_dataset, 16, sampler=valid_random_sampler)
    return Classification(train_dataloader=train_loader, val_dataloaders=valid_loader, num_classes=10, **kwargs)


def _mnist_datamodule_evaluator(**kwargs):
    class MNISTDataModule(pl.LightningDataModule):
        def __init__(self, data_dir: str = "./"):
            super().__init__()
            self.data_dir = data_dir
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        def prepare_data(self):
            # download
            MNIST(self.data_dir, train=True, download=False)
            MNIST(self.data_dir, train=False, download=False)

        def setup(self, stage):
            self.mnist_train = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_val = MNIST(self.data_dir, train=False, transform=self.transform)

            self.mnist_train = Subset(self.mnist_train, np.random.permutation(len(self.mnist_train))[:200])
            self.mnist_val = Subset(self.mnist_val, np.random.permutation(len(self.mnist_val))[:200])

        def train_dataloader(self):
            return DataLoader(self.mnist_train, batch_size=32)

        def val_dataloader(self):
            return DataLoader(self.mnist_val, batch_size=32)

    return Classification(datamodule=MNISTDataModule('data/mnist'), num_classes=10, **kwargs)


def _mhattn_evaluator(**kwargs):
    class AttentionRandDataset(Dataset):
        def __init__(self, data_shape, gt_shape, len) -> None:
            super().__init__()
            self.datashape = data_shape
            self.gtshape = gt_shape
            self.len = len

        def __getitem__(self, index):
            q = torch.rand(self.datashape)
            k = torch.rand(self.datashape)
            v = torch.rand(self.datashape)
            gt = torch.rand(self.gtshape)
            return (q, k, v), gt

        def __len__(self):
            return self.len

    train_set = AttentionRandDataset((1, 128), (1, 1), 1000)
    val_set = AttentionRandDataset((1, 128), (1, 1), 500)
    train_loader = DataLoader(train_set, batch_size=32)
    val_loader = DataLoader(val_set, batch_size=32)

    return Regression(train_dataloader=train_loader, val_dataloaders=val_loader, **kwargs)


def _model_and_evaluator(name, multi_gpu, max_epochs=1, datamodule=False):
    evaluator_kwargs = {
        'max_epochs': max_epochs
    }
    if multi_gpu:
        evaluator_kwargs.update(
            strategy='ddp',
            accelerator='gpu',
            devices=torch.cuda.device_count()
        )

    if name == 'multihead_attention':
        if datamodule:
            pytest.skip('Multi-head attention does not have datamodule evaluator.')
        evaluator = _mhattn_evaluator(**evaluator_kwargs)
    elif datamodule:
        evaluator = _mnist_datamodule_evaluator(**evaluator_kwargs)
    else:
        evaluator = _mnist_evaluator(**evaluator_kwargs)

    return MODELS[name](), evaluator


def _test_strategy(strategy, model, evaluator, expect_success):
    engine = SequentialExecutionEngine()
    if expect_success:
        # Sanity check
        strategy(RawFormatModelSpace(model, evaluator), engine)
        assert len(list(strategy.list_models())) == 1
        assert strategy.state_dict().get('ckpt_path')
    else:
        with pytest.raises(TypeError, match='not supported'):
            strategy(RawFormatModelSpace(model, evaluator), engine)


model_list = [m for m in MODELS.keys() if not m.startswith('cell_')]
model_list_if_support_value_choice = [(name, name != 'custom_op') for name in model_list]
model_list_if_not_support_value_choice = [(name, name == 'simple') for name in model_list]


@pytest.mark.parametrize('name, expect_success', model_list_if_support_value_choice)
@pytest.mark.parametrize('multi_gpu', [False, True])
@pytest.mark.parametrize('datamodule', [False, True])
def test_darts(name, expect_success, multi_gpu, datamodule):
    if multi_gpu and (torch.cuda.is_available() or torch.cuda.device_count() <= 1):
        pytest.skip('Must have multiple GPUs.')
    model, evaluator = _model_and_evaluator(name, multi_gpu, datamodule=datamodule)
    darts = strategy.DARTS()
    assert repr(darts) == 'DARTS()'
    _test_strategy(darts, model, evaluator, expect_success)


@pytest.mark.parametrize('name, expect_success', model_list_if_not_support_value_choice)
@pytest.mark.parametrize('warmup_epochs', [0, 1])
def test_proxyless(name, expect_success, warmup_epochs):
    model, evaluator = _model_and_evaluator(name, False, max_epochs=2)
    _test_strategy(strategy.Proxyless(warmup_epochs=warmup_epochs), model, evaluator, expect_success)


@pytest.mark.parametrize('name, expect_success', model_list_if_support_value_choice)
@pytest.mark.parametrize('multi_gpu', [False, True])
@pytest.mark.parametrize('warmup_epochs', [0, 1])
def test_enas(name, expect_success, multi_gpu, warmup_epochs):
    if multi_gpu and (torch.cuda.is_available() or torch.cuda.device_count() <= 1):
        pytest.skip('Must have multiple GPUs.')

    model, evaluator = _model_and_evaluator(name, multi_gpu)

    if name == 'multihead_attention':
        _test_strategy(strategy.ENAS(reward_metric_name='val_mse', warmup_epochs=warmup_epochs), model, evaluator, expect_success)
    else:
        _test_strategy(strategy.ENAS(reward_metric_name='val_acc', warmup_epochs=warmup_epochs), model, evaluator, expect_success)


@pytest.mark.parametrize('name, expect_success', model_list_if_support_value_choice)
def test_random(name, expect_success):
    model, evaluator = _model_and_evaluator(name, False)
    _test_strategy(strategy.RandomOneShot(), model, evaluator, expect_success)


@pytest.mark.parametrize('name, expect_success', model_list_if_support_value_choice)
def test_gumbel_darts(name, expect_success):
    model, evaluator = _model_and_evaluator(name, False)
    _test_strategy(strategy.GumbelDARTS(), model, evaluator, expect_success)


def test_resume(caplog):
    model, evaluator = _model_and_evaluator('simple', False)

    engine = SequentialExecutionEngine()
    strategy1 = strategy.RandomOneShot()
    strategy1(RawFormatModelSpace(model, evaluator), engine)
    state_dict = strategy1.state_dict()

    assert 'ckpt_path' in state_dict
    model, evaluator = _model_and_evaluator('simple', False, max_epochs=2)

    strategy2 = strategy.RandomOneShot()
    strategy2.initialize(RawFormatModelSpace(model, evaluator), engine)
    strategy2.load_state_dict(state_dict)
    strategy2.run()
    assert state_dict['ckpt_path'] in caplog.text


def test_optimizer_lr_scheduler():
    learning_rates = []

    class Net(ModelSpace):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(32, 2)
            self.layer2 = LayerChoice([nn.Linear(2, 2), nn.Linear(2, 2, bias=False)], label='layer')

        def forward(self, x):
            return self.layer2(self.layer1(x))

    class CustomLightningModule(LightningModule):
        def __init__(self):
            super().__init__()
            self.net = Net()

        def forward(self, x):
            return self.net(x)

        def configure_optimizers(self):
            opt1 = torch.optim.SGD(self.net.layer1.parameters(), lr=0.1)
            # no longer supported in lightning 2.x
            # opt2 = torch.optim.Adam(self.net.layer2.parameters(), lr=0.2)
            return [opt1], [torch.optim.lr_scheduler.StepLR(opt1, step_size=2, gamma=0.1)]

        def training_step(self, batch, batch_idx):
            loss = self(batch).sum()
            self.log('train_loss', loss)
            return {'loss': loss}

        def on_train_epoch_start(self) -> None:
            learning_rates.append(self.optimizers()[0].param_groups[0]['lr'])

        def validation_step(self, batch, batch_idx):
            loss = self(batch).sum()
            self.log('valid_loss', loss)

        def test_step(self, batch, batch_idx):
            loss = self(batch).sum()
            self.log('test_loss', loss)

    train_data = RandomDataset(32, 32)
    valid_data = RandomDataset(32, 16)

    model = CustomLightningModule()
    strategy.DARTS().mutate_model(model.net)
    darts_module = DartsLightningModule(model, gradient_clip_val=5)
    trainer = Trainer(max_epochs=10)
    trainer.fit(
        darts_module,
        dict(train=DataLoader(train_data, batch_size=8), val=DataLoader(valid_data, batch_size=8))
    )

    assert len(learning_rates) == 10 and abs(learning_rates[0] - 0.1) < 1e-5 and \
        abs(learning_rates[2] - 0.01) < 1e-5 and abs(learning_rates[-1] - 1e-5) < 1e-6


@pytest.mark.parametrize('name', ['simple', 'simple_value_choice', 'value_choice', 'repeat'])
def test_oneshot_freeze(name):
    x = torch.rand(1, 1, 28, 28)
    model_space = MODELS[name]()
    strategy_ = strategy.RandomOneShot()
    strategy_.mutate_model(model_space)
    oneshot_module = strategy_.configure_oneshot_module(ClassificationModule(num_classes=10))
    oneshot_module.set_model(model_space)

    arch = oneshot_module.resample()
    frozen_model = model_space.random(memo=arch)

    frozen_model.eval()
    model_space.eval()
    assert torch.allclose(frozen_model(x), model_space(x))

    # testing model recreation
    with model_context(arch):
        new_frozen_model = MODELS[name]()
    new_frozen_model.load_state_dict(frozen_model.state_dict())
    frozen_model.eval()
    assert torch.allclose(frozen_model(x), model_space(x))
