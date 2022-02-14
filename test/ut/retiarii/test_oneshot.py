import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pytest
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data.sampler import RandomSampler

from nni.retiarii.evaluator.pytorch.lightning import Classification, DataLoader
from nni.retiarii.nn.pytorch import LayerChoice, InputChoice
from nni.retiarii.oneshot.pytorch import (ConcatenateTrainValDataLoader,
                                          DartsModule, EnasModule, SNASModule,
                                          InterleavedTrainValDataLoader,
                                          ProxylessModule, RandomSampleModule)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class Net(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = LayerChoice([
            nn.Conv2d(32, 64, 3, 1),
            DepthwiseSeparableConv(32, 64)
        ])
        self.dropout1 = nn.Dropout(.25)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout_choice = InputChoice(2, 1)
        self.fc = LayerChoice([
            nn.Sequential(
                nn.Linear(9216, 64),
                nn.ReLU(),
                nn.Linear(64, 10),
            ),
            nn.Sequential(
                nn.Linear(9216, 128),
                nn.ReLU(),
                nn.Linear(128, 10),
            ),
            nn.Sequential(
                nn.Linear(9216, 256),
                nn.ReLU(),
                nn.Linear(256, 10),
            )
        ])
        self.rpfc = nn.Linear(10, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(self.conv2(x), 2)
        x1 = torch.flatten(self.dropout1(x), 1)
        x2 = torch.flatten(self.dropout2(x), 1)
        x = self.dropout_choice([x1, x2])
        x = self.fc(x)
        x = self.rpfc(x)
        output = F.log_softmax(x, dim=1)
        return output


@pytest.mark.skipif(pl.__version__< '1.0', reason='Incompatible APIs')
def prepare_model_data():
    base_model = Net()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = MNIST('data/mnist', train = True, download=True, transform=transform)
    train_random_sampler = RandomSampler(train_dataset, True, int(len(train_dataset) / 10))
    train_loader = DataLoader(train_dataset, 64, sampler = train_random_sampler)
    valid_dataset = MNIST('data/mnist', train = False, download=True, transform=transform)
    valid_random_sampler = RandomSampler(valid_dataset, True, int(len(valid_dataset) / 10))
    valid_loader = DataLoader(valid_dataset, 64, sampler = valid_random_sampler)

    trainer_kwargs = {
        'max_epochs' : 1
    }

    return base_model, train_loader, valid_loader, trainer_kwargs


@pytest.mark.skipif(pl.__version__< '1.0', reason='Incompatible APIs')
def test_darts():
    base_model, train_loader, valid_loader, trainer_kwargs = prepare_model_data()
    cls = Classification(train_dataloader=train_loader, val_dataloaders = valid_loader, **trainer_kwargs)
    cls.module.set_model(base_model)
    darts_model = DartsModule(cls.module)
    para_loader = InterleavedTrainValDataLoader(cls.train_dataloader, cls.val_dataloaders)
    cls.trainer.fit(darts_model, para_loader)


@pytest.mark.skipif(pl.__version__< '1.0', reason='Incompatible APIs')
def test_proxyless():
    base_model, train_loader, valid_loader, trainer_kwargs = prepare_model_data()
    cls = Classification(train_dataloader=train_loader, val_dataloaders=valid_loader, **trainer_kwargs)
    cls.module.set_model(base_model)
    proxyless_model = ProxylessModule(cls.module)
    para_loader = InterleavedTrainValDataLoader(cls.train_dataloader, cls.val_dataloaders)
    cls.trainer.fit(proxyless_model, para_loader)


@pytest.mark.skipif(pl.__version__< '1.0', reason='Incompatible APIs')
def test_enas():
    base_model, train_loader, valid_loader, trainer_kwargs = prepare_model_data()
    cls = Classification(train_dataloader = train_loader, val_dataloaders=valid_loader, **trainer_kwargs)
    cls.module.set_model(base_model)
    enas_model = EnasModule(cls.module)
    concat_loader = ConcatenateTrainValDataLoader(cls.train_dataloader, cls.val_dataloaders)
    cls.trainer.fit(enas_model, concat_loader)


@pytest.mark.skipif(pl.__version__< '1.0', reason='Incompatible APIs')
def test_random():
    base_model, train_loader, valid_loader, trainer_kwargs = prepare_model_data()
    cls = Classification(train_dataloader = train_loader, val_dataloaders=valid_loader , **trainer_kwargs)
    cls.module.set_model(base_model)
    random_model = RandomSampleModule(cls.module)
    cls.trainer.fit(random_model, cls.train_dataloader, cls.val_dataloaders)


@pytest.mark.skipif(pl.__version__< '1.0', reason='Incompatible APIs')
def test_snas():
    base_model, train_loader, valid_loader, trainer_kwargs = prepare_model_data()
    cls = Classification(train_dataloader=train_loader, val_dataloaders=valid_loader, **trainer_kwargs)
    cls.module.set_model(base_model)
    proxyless_model = SNASModule(cls.module, 1, use_temp_anneal=True)
    para_loader = InterleavedTrainValDataLoader(cls.train_dataloader, cls.val_dataloaders)
    cls.trainer.fit(proxyless_model, para_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='all', metavar='E',
        help='exp to run, default = all' )
    args = parser.parse_args()

    if args.exp == 'all':
        test_darts()
        test_proxyless()
        test_enas()
        test_random()
        test_snas()
    else:
        globals()[f'test_{args.exp}']()
