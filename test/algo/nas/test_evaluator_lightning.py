import pytest

import pytorch_lightning
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_diabetes
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST

import nni
import nni.nas.evaluator.pytorch.lightning as pl
from nni.nas.evaluator import FunctionalEvaluator, Evaluator
from nni.nas.space import RawFormatModelSpace

debug = True

enable_progress_bar = False
if debug:
    enable_progress_bar = True


@pytest.fixture
def mocked_model():
    model = RawFormatModelSpace(None, None)
    with Evaluator.mock_runtime(model):
        yield model


class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        return x


class FCNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, 5)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(5, output_size)

    def forward(self, x):
        output = self.l1(x)
        output = self.relu(output)
        output = self.l2(output)
        return output.view(-1)


@nni.trace
class DiabetesDataset(Dataset):
    def __init__(self, train=True):
        data = load_diabetes()
        self.x = torch.tensor(data['data'], dtype=torch.float32)
        self.y = torch.tensor(data['target'], dtype=torch.float32)
        self.length = self.x.shape[0]
        split = int(self.length * 0.8)
        if train:
            self.x = self.x[:split]
            self.y = self.y[:split]
        else:
            self.x = self.x[split:]
            self.y = self.y[split:]
        self.length = len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.length


def _foo(model):
    assert isinstance(model, MNISTModel)


@pytest.mark.skipif(pytorch_lightning.__version__ < '1.0', reason='Incompatible APIs.')
def test_mnist(mocked_model):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = nni.trace(MNIST)(root='data/mnist', train=True, download=True, transform=transform)
    test_dataset = nni.trace(MNIST)(root='data/mnist', train=False, download=True, transform=transform)
    lightning = pl.Classification(train_dataloader=pl.DataLoader(train_dataset, batch_size=100),
                                  val_dataloaders=pl.DataLoader(test_dataset, batch_size=100),
                                  max_epochs=2, limit_train_batches=0.25,  # for faster training
                                  enable_progress_bar=enable_progress_bar,
                                  num_classes=10)
    lightning.evaluate(MNISTModel())
    assert mocked_model.metric > 0.7
    assert len(mocked_model.metrics.intermediates) == 2


@pytest.mark.skipif(pytorch_lightning.__version__ < '1.0', reason='Incompatible APIs.')
def test_diabetes(mocked_model):
    train_dataset = DiabetesDataset(train=True)
    test_dataset = DiabetesDataset(train=False)
    lightning = pl.Regression(optimizer=torch.optim.SGD,
                              train_dataloader=pl.DataLoader(train_dataset, batch_size=20),
                              val_dataloaders=pl.DataLoader(test_dataset, batch_size=20),
                              max_epochs=100,
                              enable_progress_bar=enable_progress_bar)
    lightning.evaluate(FCNet(train_dataset.x.shape[1], 1))
    assert mocked_model.metric < 2e4
    assert len(mocked_model.metrics.intermediates) == 100


@pytest.mark.skipif(pytorch_lightning.__version__ < '1.0', reason='Incompatible APIs.')
def test_functional():
    FunctionalEvaluator(_foo).evaluate(MNISTModel())


@pytest.mark.skipif(pytorch_lightning.__version__ < '1.0', reason='Incompatible APIs.')
def test_fit_api(mocked_model):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = nni.trace(MNIST)(root='data/mnist', train=True, download=True, transform=transform)
    test_dataset = nni.trace(MNIST)(root='data/mnist', train=False, download=True, transform=transform)

    def lightning(): return pl.Classification(train_dataloader=pl.DataLoader(train_dataset, batch_size=100),
                                              val_dataloaders=pl.DataLoader(test_dataset, batch_size=100),
                                              max_epochs=1, limit_train_batches=0.1,  # for faster training
                                              enable_progress_bar=enable_progress_bar,
                                              num_classes=10)
    # Lightning will have some cache in models / trainers,
    # which is problematic if we call fit multiple times.
    lightning().evaluate(MNISTModel())
