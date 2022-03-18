import argparse
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import pytest
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data.sampler import RandomSampler

import nni.retiarii.nn.pytorch as nn
from nni.retiarii import strategy, model_wrapper
from nni.retiarii.experiment.pytorch import RetiariiExeConfig, RetiariiExperiment
from nni.retiarii.evaluator.pytorch.lightning import Classification, DataLoader
from nni.retiarii.nn.pytorch import LayerChoice, InputChoice, ValueChoice


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


@model_wrapper
class Net(nn.Module):
    def __init__(self, value_choice=True):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = LayerChoice([
            nn.Conv2d(32, 64, 3, 1),
            DepthwiseSeparableConv(32, 64)
        ])
        self.dropout1 = LayerChoice([
            nn.Dropout(.25),
            nn.Dropout(.5),
            nn.Dropout(.75)
        ])
        self.dropout2 = nn.Dropout(0.5)
        if value_choice:
            hidden = nn.ValueChoice([64, 128, 196])
        else:
            hidden = 128
        self.fc1 = nn.Linear(9216, hidden)
        self.fc2 = nn.Linear(hidden, 10)
        self.rpfc = nn.Linear(10, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(self.conv2(x), 2)
        x = torch.flatten(self.dropout1(x), 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.rpfc(x)
        output = F.log_softmax(x, dim=1)
        return output


def _test_strategy(strategy_, support_value_choice=True):
    base_model = Net(support_value_choice)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = MNIST('data/mnist', train=True, download=True, transform=transform)
    train_random_sampler = RandomSampler(train_dataset, True, int(len(train_dataset) / 10))
    train_loader = DataLoader(train_dataset, 64, sampler=train_random_sampler)
    valid_dataset = MNIST('data/mnist', train=False, download=True, transform=transform)
    valid_random_sampler = RandomSampler(valid_dataset, True, int(len(valid_dataset) / 10))
    valid_loader = DataLoader(valid_dataset, 64, sampler=valid_random_sampler)

    trainer_kwargs = {
        'max_epochs': 1
    }

    cls = Classification(train_dataloader=train_loader, val_dataloaders=valid_loader, **trainer_kwargs)
    experiment = RetiariiExperiment(base_model, cls, strategy=strategy_)

    config = RetiariiExeConfig()
    config.execution_engine = 'oneshot'

    experiment.run(config)

    assert isinstance(experiment.export_top_models()[0], dict)


@pytest.mark.skipif(pl.__version__ < '1.0', reason='Incompatible APIs')
def test_darts():
    _test_strategy(strategy.DARTS())


@pytest.mark.skipif(pl.__version__ < '1.0', reason='Incompatible APIs')
def test_proxyless():
    _test_strategy(strategy.Proxyless(), False)


@pytest.mark.skipif(pl.__version__ < '1.0', reason='Incompatible APIs')
def test_enas():
    _test_strategy(strategy.ENAS())


@pytest.mark.skipif(pl.__version__ < '1.0', reason='Incompatible APIs')
def test_random():
    _test_strategy(strategy.RandomOneShot())


@pytest.mark.skipif(pl.__version__ < '1.0', reason='Incompatible APIs')
def test_gumbel_darts():
    _test_strategy(strategy.GumbelDARTS())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='all', metavar='E',
                        help='experiment to run, default = all')
    args = parser.parse_args()

    if args.exp == 'all':
        test_darts()
        test_proxyless()
        test_enas()
        test_random()
        test_gumbel_darts()
    else:
        globals()[f'test_{args.exp}']()
