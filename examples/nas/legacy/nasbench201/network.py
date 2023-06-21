import click
import nni
import nni.retiarii.evaluator.pytorch.lightning as pl
import torch.nn as nn
import torchmetrics
from nni.retiarii import model_wrapper, serialize
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
from nni.retiarii.nn.pytorch import NasBench201Cell
from nni.retiarii.strategy import Random
from pytorch_lightning.callbacks import LearningRateMonitor
from timm.optim import RMSpropTF
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torchvision.datasets import CIFAR100

from base_ops import ResNetBasicblock, PRIMITIVES, OPS_WITH_STRIDE


@model_wrapper
class NasBench201(nn.Module):
    def __init__(self,
                 stem_out_channels: int = 16,
                 num_modules_per_stack: int = 5,
                 num_labels: int = 100):
        super().__init__()
        self.channels = C = stem_out_channels
        self.num_modules = N = num_modules_per_stack
        self.num_labels = num_labels

        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C)
        )

        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        C_prev = C
        self.cells = nn.ModuleList()
        for C_curr, reduction in zip(layer_channels, layer_reductions):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2)
            else:
                cell = NasBench201Cell({prim: lambda C_in, C_out: OPS_WITH_STRIDE[prim](C_in, C_out, 1) for prim in PRIMITIVES},
                                       C_prev, C_curr, label='cell')
            self.cells.append(cell)
            C_prev = C_curr

        self.lastact = nn.Sequential(
            nn.BatchNorm2d(C_prev),
            nn.ReLU(inplace=True)
        )
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, self.num_labels)

    def forward(self, inputs):
        feature = self.stem(inputs)
        for cell in self.cells:
            feature = cell(feature)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return logits


class AccuracyWithLogits(torchmetrics.Accuracy):
    def update(self, pred, target):
        return super().update(nn.functional.softmax(pred), target)


@nni.trace
class NasBench201TrainingModule(pl.LightningModule):
    def __init__(self, max_epochs=200, learning_rate=0.1, weight_decay=5e-4):
        super().__init__()
        self.save_hyperparameters('learning_rate', 'weight_decay', 'max_epochs')
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = AccuracyWithLogits()

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_accuracy', self.accuracy(y_hat, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.log('val_loss', self.criterion(y_hat, y), prog_bar=True)
        self.log('val_accuracy', self.accuracy(y_hat, y), prog_bar=True)

    def configure_optimizers(self):
        optimizer = RMSpropTF(self.parameters(), lr=self.hparams.learning_rate,
                              weight_decay=self.hparams.weight_decay,
                              momentum=0.9, alpha=0.9, eps=1.0)
        return {
            'optimizer': optimizer,
            'lr_scheduler': CosineAnnealingLR(optimizer, self.hparams.max_epochs)
        }

    def on_validation_epoch_end(self):
        nni.report_intermediate_result(self.trainer.callback_metrics['val_accuracy'].item())

    def teardown(self, stage):
        if stage == 'fit':
            nni.report_final_result(self.trainer.callback_metrics['val_accuracy'].item())


@click.command()
@click.option('--epochs', default=12, help='Training length.')
@click.option('--batch_size', default=256, help='Batch size.')
@click.option('--port', default=8081, help='On which port the experiment is run.')
@click.option('--benchmark', is_flag=True, default=False)
def _multi_trial_test(epochs, batch_size, port, benchmark):
    # initalize dataset. Note that 50k+10k is used. It's a little different from paper
    transf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize([x / 255 for x in [129.3, 124.1, 112.4]], [x / 255 for x in [68.2, 65.4, 70.4]])
    ]
    train_dataset = serialize(CIFAR100, 'data', train=True, download=True, transform=transforms.Compose(transf + normalize))
    test_dataset = serialize(CIFAR100, 'data', train=False, transform=transforms.Compose(normalize))

    # specify training hyper-parameters
    training_module = NasBench201TrainingModule(max_epochs=epochs)
    # FIXME: need to fix a bug in serializer for this to work
    # lr_monitor = serialize(LearningRateMonitor, logging_interval='step')
    trainer = pl.Trainer(max_epochs=epochs, gpus=1)
    lightning = pl.Lightning(
        lightning_module=training_module,
        trainer=trainer,
        train_dataloader=pl.DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        val_dataloaders=pl.DataLoader(test_dataset, batch_size=batch_size),
    )

    strategy = Random()

    model = NasBench201()

    exp = RetiariiExperiment(model, lightning, [], strategy)

    exp_config = RetiariiExeConfig('local')
    exp_config.trial_concurrency = 2
    exp_config.max_trial_number = 20
    exp_config.trial_gpu_number = 1
    exp_config.training_service.use_active_gpu = False

    if benchmark:
        exp_config.benchmark = 'nasbench201-cifar100'
        exp_config.execution_engine = 'benchmark'

    exp.run(exp_config, port)


if __name__ == '__main__':
    _multi_trial_test()
