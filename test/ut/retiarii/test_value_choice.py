import argparse

import nni.retiarii.nn.pytorch.nn as nn
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.data as data
from nni.retiarii.evaluator.pytorch.lightning import (Classification,
                                                      DataLoader, Regression)
from nni.retiarii.nn.pytorch import LayerChoice, ValueChoice
from nni.retiarii.oneshot.pytorch import (ConcatenateTrainValDataLoader,
                                          DartsModule, EnasModule,
                                          ParallelTrainValDataLoader,
                                          RandomSampleModule)
from torchvision import transforms
from torchvision.datasets import MNIST

import pytest


class TestMultiHeadAttentionNet(nn.Module):
    def __init__(self, embed_dim, head_count):
        super().__init__()
        self.linear1 = nn.Linear(128, embed_dim)
        self.mhatt = nn.MultiheadAttention(embed_dim, head_count)
        self.linear2 = nn.Linear(embed_dim, 1)
    
    def forward(self, batch):
        query, key, value = batch
        q, k, v = self.linear1(query), self.linear1(key), self.linear1(value)
        output, _ = self.mhatt(q, k, v, need_weights = False)
        y = self.linear2(output)
        return F.relu(y)


class Net(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, ValueChoice(candidates = [16, 32], label='shandong'), 3)
        self.batch_norm = nn.BatchNorm2d(ValueChoice(candidates=[16, 32], label='shandong'))
        self.conv2 = nn.Conv2d(ValueChoice(candidates=[16, 32], label='shandong'), 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 
            kernel_size = ValueChoice([3, 5], label='beijing'),
            padding = ValueChoice([1, 2], label = 'beijing')
            )
        self.dropout1 = LayerChoice([
            nn.Dropout(.25),
            nn.Dropout(.5),
            nn.Dropout(.75)
        ])
        self.fc1 = nn.Linear(9216, ValueChoice(candidates=[64,128,256], label = 'shanghai'))
        self.fc2 = nn.Linear(ValueChoice(candidates=[64,128,256], label = 'shanghai'), 10)
        self.rpfc = nn.Linear(10, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = F.max_pool2d(self.conv2(x), 2)
        x = self.conv3(x)
        x = torch.flatten(self.dropout1(x), 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.rpfc(x)
        output = F.log_softmax(x, dim=1)
        return output

class DartsConv2dValueChoice(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, ValueChoice(candidates=[16, 32], label = 'fudan'), ValueChoice(candidates=[3, 5], label = 'ecnu'), padding = 1)
        self.batch_norm = nn.BatchNorm2d(ValueChoice(candidates=[16, 32], label = 'fudan'))
        self.conv2 = nn.Conv2d(ValueChoice(candidates=[16, 32], label = 'fudan'), 64, 3)
        self.dropout1 = LayerChoice([
            nn.Dropout(.25),
            nn.Dropout(.5),
            nn.Dropout(.75)
        ])
        self.fc1 = nn.Linear(9216, 256)
        self.fc2 = nn.Linear(256, 10)
        self.rpfc = nn.Linear(10, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = F.max_pool2d(self.conv2(x), 2)
        x = torch.flatten(self.dropout1(x), 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.rpfc(x)
        output = F.log_softmax(x, dim=1)
        return output


@pytest.mark.skipif(pl.__version__< '1.0', reason='Incompatible APIs')
def prepare_model_data():
    base_model = Net()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = MNIST('data/mnist', train = True, download=True, transform=transform)
    train_random_sampler = data.RandomSampler(train_dataset, True, int(len(train_dataset) / 10))
    train_loader = DataLoader(train_dataset, 64, sampler = train_random_sampler)
    valid_dataset = MNIST('data/mnist', train = False, download=True, transform=transform)
    valid_random_sampler = data.RandomSampler(valid_dataset, True, int(len(valid_dataset) / 10))
    valid_loader = DataLoader(valid_dataset, 64, sampler = valid_random_sampler)

    trainer_kwargs = {
        'max_epochs' : 1
    }

    return base_model, train_loader, valid_loader, trainer_kwargs


def prepare_multiheadattention_data():
    base_model = TestMultiHeadAttentionNet(ValueChoice(candidates=[32, 64], label='tianjin'), 1)
    class AttentionRandDataset(data.Dataset):
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
    
    train_set = AttentionRandDataset((1,128), (1,1), 1000)
    val_set = AttentionRandDataset((1,128), (1, 1), 500)
    train_loader = DataLoader(train_set, 32)
    val_loader = DataLoader(val_set)
    
    trainer_kwargs={ 'max_epochs': 1 }

    return base_model, train_loader, val_loader, trainer_kwargs


@pytest.mark.skipif(pl.__version__< '1.0', reason='Incompatible APIs')
def test_darts():
    base_model, train_loader, valid_loader, trainer_kwargs = prepare_model_data()
    base_model = DartsConv2dValueChoice()
    cls = Classification(train_dataloader=train_loader, val_dataloaders = valid_loader, **trainer_kwargs)
    cls.module.set_model(base_model)
    darts_model = DartsModule(cls.module)
    para_loader = ParallelTrainValDataLoader(cls.train_dataloader, cls.val_dataloaders)
    cls.trainer.fit(darts_model, para_loader)
    

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
def test_mhatt():
    base_model, train_loader, valid_loader, trainer_kwargs = prepare_multiheadattention_data()
    reg = Regression(train_dataloader = train_loader, val_dataloaders=valid_loader , **trainer_kwargs)
    reg.module.set_model(base_model)
    random_model = RandomSampleModule(reg.module)
    reg.trainer.fit(random_model, reg.train_dataloader, reg.val_dataloaders)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='all', metavar='E',
        help='experiment to run, default = all' )
    args = parser.parse_args()

    if args.exp == 'all':
        test_darts()
        test_enas()
        test_random()
        test_mhatt()
    else:
        globals()[f'test_{args.exp}']()
