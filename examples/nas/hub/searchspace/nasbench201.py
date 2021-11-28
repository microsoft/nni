import torch
import torch.nn as nn

import click
import nni
import nni.retiarii.evaluator.pytorch.lightning as pl
import torch.nn as nn
import torchmetrics
from nni.retiarii import model_wrapper, serialize, serialize_cls
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
from nni.retiarii.nn.pytorch import NasBench201Cell
from nni.retiarii.strategy import Random
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torchvision.datasets import CIFAR100


OPS_WITH_STRIDE = {
    'none': lambda C_in, C_out, stride: Zero(C_in, C_out, stride),
    'avg_pool_3x3': lambda C_in, C_out, stride: Pooling(C_in, C_out, stride, 'avg'),
    'max_pool_3x3': lambda C_in, C_out, stride: Pooling(C_in, C_out, stride, 'max'),
    'conv_3x3': lambda C_in, C_out, stride: ReLUConvBN(C_in, C_out, (3, 3), (stride, stride), (1, 1), (1, 1)),
    'conv_1x1': lambda C_in, C_out, stride: ReLUConvBN(C_in, C_out, (1, 1), (stride, stride), (0, 0), (1, 1)),
    'skip_connect': lambda C_in, C_out, stride: nn.Identity() if stride == 1 and C_in == C_out
    else FactorizedReduce(C_in, C_out, stride),
}

PRIMITIVES = ['none', 'skip_connect', 'conv_1x1', 'conv_3x3', 'avg_pool_3x3']


class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(C_out)
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out),
        )

    def forward(self, x):
        return self.op(x)


class Pooling(nn.Module):
    def __init__(self, C_in, C_out, stride, mode):
        super(Pooling, self).__init__()
        if C_in == C_out:
            self.preprocess = None
        else:
            self.preprocess = ReLUConvBN(C_in, C_out, 1, 1, 0, 1)
        if mode == 'avg':
            self.op = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
        elif mode == 'max':
            self.op = nn.MaxPool2d(3, stride=stride, padding=1)
        else:
            raise ValueError('Invalid mode={:} in Pooling'.format(mode))

    def forward(self, x):
        if self.preprocess:
            x = self.preprocess(x)
        return self.op(x)


class Zero(nn.Module):
    def __init__(self, C_in, C_out, stride):
        super(Zero, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.is_zero = True

    def forward(self, x):
        if self.C_in == self.C_out:
            if self.stride == 1:
                return x.mul(0.)
            else:
                return x[:, :, ::self.stride, ::self.stride].mul(0.)
        else:
            shape = list(x.shape)
            shape[1] = self.C_out
            zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
            return zeros


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, stride):
        super(FactorizedReduce, self).__init__()
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        self.relu = nn.ReLU(inplace=False)
        if stride == 2:
            C_outs = [C_out // 2, C_out - C_out // 2]
            self.convs = nn.ModuleList()
            for i in range(2):
                self.convs.append(nn.Conv2d(C_in, C_outs[i], 1, stride=stride, padding=0, bias=False))
            self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        else:
            raise ValueError('Invalid stride : {:}'.format(stride))
        self.bn = nn.BatchNorm2d(C_out)

    def forward(self, x):
        x = self.relu(x)
        y = self.pad(x)
        out = torch.cat([self.convs[0](x), self.convs[1](y[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class ResNetBasicblock(nn.Module):
    def __init__(self, inplanes, planes, stride):
        super(ResNetBasicblock, self).__init__()
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        self.conv_a = ReLUConvBN(inplanes, planes, 3, stride, 1, 1)
        self.conv_b = ReLUConvBN(planes, planes, 3, 1, 1, 1)
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False))
        elif inplanes != planes:
            self.downsample = ReLUConvBN(inplanes, planes, 1, 1, 0, 1)
        else:
            self.downsample = None
        self.in_dim = inplanes
        self.out_dim = planes
        self.stride = stride
        self.num_conv = 2

    def forward(self, inputs):
        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)

        if self.downsample is not None:
            inputs = self.downsample(inputs)  # residual
        return inputs + basicblock




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


@serialize_cls
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
            'scheduler': CosineAnnealingLR(optimizer, self.hparams.max_epochs)
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
