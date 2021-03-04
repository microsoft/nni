import json
import pytest

import nni
import nni.retiarii.evaluator.pytorch.lightning as pl
import pytorch_lightning
import torch
import torch.nn as nn
import torch.nn.functional as F
from nni.retiarii import serialize_cls, serialize
from nni.retiarii.evaluator import FunctionalEvaluator
from sklearn.datasets import load_diabetes
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST

debug = False

progress_bar_refresh_rate = 0
if debug:
    progress_bar_refresh_rate = 1


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


@serialize_cls
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


def _get_final_result():
    return float(json.loads(nni.runtime.platform.test._last_metric)['value'])


def _foo(model_cls):
    assert model_cls == MNISTModel


def _reset():
    # this is to not affect other tests in sdk
    nni.trial._intermediate_seq = 0
    nni.trial._params = {'foo': 'bar', 'parameter_id': 0}
    nni.runtime.platform.test._last_metric = None


@pytest.mark.skipif(pytorch_lightning.__version__ < '1.0', reason='Incompatible APIs.')
def test_mnist():
    _reset()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = serialize(MNIST, root='data/mnist', train=True, download=True, transform=transform)
    test_dataset = serialize(MNIST, root='data/mnist', train=False, download=True, transform=transform)
    lightning = pl.Classification(train_dataloader=pl.DataLoader(train_dataset, batch_size=100),
                                  val_dataloaders=pl.DataLoader(test_dataset, batch_size=100),
                                  max_epochs=2, limit_train_batches=0.25,  # for faster training
                                  progress_bar_refresh_rate=progress_bar_refresh_rate)
    lightning._execute(MNISTModel)
    assert _get_final_result() > 0.7
    _reset()


@pytest.mark.skipif(pytorch_lightning.__version__ < '1.0', reason='Incompatible APIs.')
def test_diabetes():
    _reset()
    nni.trial._params = {'foo': 'bar', 'parameter_id': 0}
    nni.runtime.platform.test._last_metric = None
    train_dataset = DiabetesDataset(train=True)
    test_dataset = DiabetesDataset(train=False)
    lightning = pl.Regression(optimizer=torch.optim.SGD,
                              train_dataloader=pl.DataLoader(train_dataset, batch_size=20),
                              val_dataloaders=pl.DataLoader(test_dataset, batch_size=20),
                              max_epochs=100,
                              progress_bar_refresh_rate=progress_bar_refresh_rate)
    lightning._execute(FCNet(train_dataset.x.shape[1], 1))
    assert _get_final_result() < 2e4
    _reset()


@pytest.mark.skipif(pytorch_lightning.__version__ < '1.0', reason='Incompatible APIs.')
def test_functional():
    FunctionalEvaluator(_foo)._execute(MNISTModel)


if __name__ == '__main__':
    test_mnist()
    test_diabetes()
    test_functional()
