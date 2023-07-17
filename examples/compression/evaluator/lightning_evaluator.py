"""
Create Lightning Evaluator
==========================

To create a lightning evaluator in NNI,
you only need to make minor modifications to your existing code.

Modificatoin in LightningModule
-------------------------------

In ``configure_optimizers``, please using ``nni.trace`` to trace the optimizer and lr scheduler class.

Please set a ``default`` metric in ``validation_step`` or ``test_step`` if it needs,
NNI may use this metric to compare which model is better.

"""
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torchmetrics.functional import accuracy

import nni
from examples.compression.models import build_resnet18


class MyModule(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = build_resnet18()
        self.criterion = torch.nn.CrossEntropyLoss()

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

        # If NNI need to evaluate the model, "default" metric will be used.
        if stage:
            self.log(f"default", loss, prog_bar=False)
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = nni.trace(torch.optim.Adam)(self.parameters(), lr=0.001)
        scheduler_dict = {
            "scheduler": nni.trace(torch.optim.lr_scheduler.LambdaLR)(optimizer, lr_lambda=lambda epoch: 1 / epoch),
            "interval": "epoch",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


# %%
# Init ``TorchEvaluator``
# -----------------------
#
# Remember using ``nni.trace`` to trace ``Trainer`` and your customized ``LightningDataModule``.

# directly using your original LightningDataModule
class MyDataModule(pl.LightningDataModule):
    pass

from nni.compression import LightningEvaluator

pl_trainer = nni.trace(pl.Trainer)(
    accelerator='auto',
    devices=1,
    max_epochs=3,
    logger=TensorBoardLogger('./lightning_logs', name="vgg"),
)
pl_data = nni.trace(MyDataModule)(data_dir='./data')

evaluator = LightningEvaluator(pl_trainer, pl_data)
