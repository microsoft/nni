# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations
import math

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
from torchvision import datasets, transforms

import nni
from nni.contrib.compression.utils import LightningEvaluator
from nni.contrib.compression.quantization import LsqQuantizer


defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


class VGG(nn.Module):
    def __init__(self, depth=16):
        super(VGG, self).__init__()
        cfg = defaultcfg[depth]
        self.cfg = cfg
        self.feature = self.make_layers(cfg, True)
        num_classes = 10
        self.classifier = nn.Sequential(
            nn.Linear(cfg[-1], 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )
        self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class SimpleLightningModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = VGG()
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, 'multiclass', num_classes=10)

        if stage:
            self.log(f"default", loss, prog_bar=False)
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = nni.trace(torch.optim.Adam)(
            self.parameters(),
            lr=0.001
        )
        scheduler_dict = {
            "scheduler": nni.trace(StepLR)(
                optimizer,
                step_size=1,
                gamma=0.5
            ),
            "interval": "epoch",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data"):
        super().__init__()
        self.data_dir = data_dir

    def prepare_data(self):
        # download
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            self.cifar10_train_data = datasets.CIFAR10(root='data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]))
            self.cifar10_val_data = datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]))

        if stage == "test" or stage is None:
            self.cifar10_test_data = datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]))

        if stage == "predict" or stage is None:
            self.cifar10_predict_data = datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]))

    def train_dataloader(self):
        return DataLoader(self.cifar10_train_data, batch_size=128, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cifar10_val_data, batch_size=128, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.cifar10_test_data, batch_size=128, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.cifar10_predict_data, batch_size=128, shuffle=False)


# Train the model
pl_trainer = nni.trace(pl.Trainer)(
    accelerator='auto',
    devices=1,
    max_epochs=3,
    logger=TensorBoardLogger('./lightning_logs', name="vgg"),
)
pl_data = nni.trace(ImageNetDataModule)(data_dir='./data')
model = SimpleLightningModel()
pl_trainer.fit(model, pl_data)
metric = pl_trainer.test(model, pl_data)
print(f'The trained model accuracy: {metric}')

# create traced optimizer / lr_scheduler
optimizer = nni.trace(torch.optim.Adam)(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()
lr_scheduler = nni.trace(StepLR)(optimizer, step_size=1, gamma=0.5)
dummy_input = torch.rand(4, 3, 224, 224)

# TorchEvaluator initialization
evaluator = LightningEvaluator(pl_trainer, pl_data)
configure_list = [{
        'quant_types': ['weight'],
        'quant_bits': 8,
        'op_types': ['Conv2d', 'Linear'],
        'op_names': ['model.feature.3', 'model.feature.7', 'model.feature.10', 'model.feature.14', 'model.classifier.0', 'model.classifier.3']
    }, {
        'quant_types': ['output'],
        'quant_bits': 8,
        'op_types': ['ReLU'],
        'op_names': ['model.feature.2', 'model.feature.5']
    }]

metric = pl_trainer.test(model, pl_data)
print(f'metric: {metric}')
quantizer = LsqQuantizer(model, configure_list, evaluator)
quantizer.compress()