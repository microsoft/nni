import click
import nni
import nni.retiarii.evaluator.pytorch.lightning as pl
import torch.nn as nn
import torchmetrics
from nni.retiarii import model_wrapper, serialize
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
from nni.retiarii.nn.pytorch import NasBench101Cell
from nni.retiarii.strategy import Random
from pytorch_lightning.callbacks import LearningRateMonitor
from timm.optim import RMSpropTF
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torchvision.datasets import CIFAR10

from base_ops import Conv3x3BnRelu, Conv1x1BnRelu, Projection


@model_wrapper
class NasBench101(nn.Module):
    def __init__(self,
                 stem_out_channels: int = 128,
                 num_stacks: int = 3,
                 num_modules_per_stack: int = 3,
                 max_num_vertices: int = 7,
                 max_num_edges: int = 9,
                 num_labels: int = 10,
                 bn_eps: float = 1e-5,
                 bn_momentum: float = 0.003):
        super().__init__()

        op_candidates = {
            'conv3x3-bn-relu': lambda num_features: Conv3x3BnRelu(num_features, num_features),
            'conv1x1-bn-relu': lambda num_features: Conv1x1BnRelu(num_features, num_features),
            'maxpool3x3': lambda num_features: nn.MaxPool2d(3, 1, 1)
        }

        # initial stem convolution
        self.stem_conv = Conv3x3BnRelu(3, stem_out_channels)

        layers = []
        in_channels = out_channels = stem_out_channels
        for stack_num in range(num_stacks):
            if stack_num > 0:
                downsample = nn.MaxPool2d(kernel_size=2, stride=2)
                layers.append(downsample)
                out_channels *= 2
            for _ in range(num_modules_per_stack):
                cell = NasBench101Cell(op_candidates, in_channels, out_channels,
                                       lambda cin, cout: Projection(cin, cout),
                                       max_num_vertices, max_num_edges, label='cell')
                layers.append(cell)
                in_channels = out_channels

        self.features = nn.ModuleList(layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(out_channels, num_labels)

        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eps = bn_eps
                module.momentum = bn_momentum

    def forward(self, x):
        bs = x.size(0)
        out = self.stem_conv(x)
        for layer in self.features:
            out = layer(out)
        out = self.gap(out).view(bs, -1)
        out = self.classifier(out)
        return out

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eps = self.config.bn_eps
                module.momentum = self.config.bn_momentum


class AccuracyWithLogits(torchmetrics.Accuracy):
    def update(self, pred, target):
        return super().update(nn.functional.softmax(pred), target)


@nni.trace
class NasBench101TrainingModule(pl.LightningModule):
    def __init__(self, max_epochs=108, learning_rate=0.1, weight_decay=1e-4):
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
@click.option('--epochs', default=108, help='Training length.')
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
        transforms.Normalize([0.49139968, 0.48215827, 0.44653124], [0.24703233, 0.24348505, 0.26158768])
    ]
    train_dataset = serialize(CIFAR10, 'data', train=True, download=True, transform=transforms.Compose(transf + normalize))
    test_dataset = serialize(CIFAR10, 'data', train=False, transform=transforms.Compose(normalize))

    # specify training hyper-parameters
    training_module = NasBench101TrainingModule(max_epochs=epochs)
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

    model = NasBench101()

    exp = RetiariiExperiment(model, lightning, [], strategy)

    exp_config = RetiariiExeConfig('local')
    exp_config.trial_concurrency = 2
    exp_config.max_trial_number = 20
    exp_config.trial_gpu_number = 1
    exp_config.training_service.use_active_gpu = False

    if benchmark:
        exp_config.benchmark = 'nasbench101'
        exp_config.execution_engine = 'benchmark'

    exp.run(exp_config, port)


if __name__ == '__main__':
    _multi_trial_test()
