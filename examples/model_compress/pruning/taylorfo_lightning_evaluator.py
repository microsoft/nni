from __future__ import annotations
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
from torchvision import datasets, transforms

import nni
from nni.compression.pytorch import LightningEvaluator

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[1] / 'models'))
from cifar10.vgg import VGG


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
        acc = accuracy(preds, y)

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

# apply pruning
from nni.compression.pytorch.pruning import TaylorFOWeightPruner
from nni.compression.pytorch.speedup import ModelSpeedup

pruner = TaylorFOWeightPruner(model, config_list=[{'total_sparsity': 0.5, 'op_types': ['Conv2d']}], evaluator=evaluator, training_steps=100)
_, masks = pruner.compress()
metric = pl_trainer.test(model, pl_data)
print(f'The masked model accuracy: {metric}')
pruner.show_pruned_weights()
pruner._unwrap_model()
ModelSpeedup(model, dummy_input=torch.rand([10, 3, 32, 32]), masks_file=masks).speedup_model()
metric = pl_trainer.test(model, pl_data)
print(f'The speedup model accuracy: {metric}')

# finetune the speedup model
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()
lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

pl_trainer = pl.Trainer(
    accelerator='auto',
    devices=1,
    max_epochs=3,
    logger=TensorBoardLogger('./lightning_logs', name="vgg"),
)
pl_trainer.fit(model, pl_data)
metric = pl_trainer.test(model, pl_data)
print(f'The speedup model after finetuning accuracy: {metric}')
