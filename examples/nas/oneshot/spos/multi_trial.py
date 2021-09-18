# This file is to demo the usage of multi-trial NAS in the usage of SPOS search space.

import click
import nni.retiarii.evaluator.pytorch as pl
import nni.retiarii.nn.pytorch as nn
import nni.retiarii.strategy as strategy
import torch
from nni.retiarii import serialize
from nni.retiarii.nn.pytorch import LayerChoice
from nni.retiarii.experiment.pytorch import RetiariiExeConfig, RetiariiExperiment
from torchvision import transforms
from torchvision.datasets import CIFAR10

from blocks import ShuffleNetBlock, ShuffleXceptionBlock

from nn_meter import load_latency_predictor


class ShuffleNetV2(nn.Module):
    block_keys = [
        'shufflenet_3x3',
        'shufflenet_5x5',
        'shufflenet_7x7',
        'xception_3x3',
    ]

    def __init__(self, input_size=224, first_conv_channels=16, last_conv_channels=1024, n_classes=1000, affine=False):
        super().__init__()

        assert input_size % 32 == 0

        self.stage_blocks = [4, 4, 8, 4]
        self.stage_channels = [64, 160, 320, 640]
        self._parsed_flops = dict()
        self._input_size = input_size
        self._feature_map_size = input_size
        self._first_conv_channels = first_conv_channels
        self._last_conv_channels = last_conv_channels
        self._n_classes = n_classes
        self._affine = affine

        # building first layer
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, first_conv_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(first_conv_channels, affine=affine),
            nn.ReLU(inplace=True),
        )
        self._feature_map_size //= 2

        p_channels = first_conv_channels
        features = []
        for num_blocks, channels in zip(self.stage_blocks, self.stage_channels):
            features.extend(self._make_blocks(num_blocks, p_channels, channels))
            p_channels = channels
        self.features = nn.Sequential(*features)

        self.conv_last = nn.Sequential(
            nn.Conv2d(p_channels, last_conv_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(last_conv_channels, affine=affine),
            nn.ReLU(inplace=True),
        )
        self.globalpool = nn.AvgPool2d(self._feature_map_size)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(last_conv_channels, n_classes, bias=False),
        )

        self._initialize_weights()

    def _make_blocks(self, blocks, in_channels, channels):
        result = []
        for i in range(blocks):
            stride = 2 if i == 0 else 1
            inp = in_channels if i == 0 else channels
            oup = channels

            base_mid_channels = channels // 2
            mid_channels = int(base_mid_channels)  # prepare for scale
            choice_block = LayerChoice([
                ShuffleNetBlock(inp, oup, mid_channels=mid_channels, ksize=3, stride=stride, affine=self._affine),
                ShuffleNetBlock(inp, oup, mid_channels=mid_channels, ksize=5, stride=stride, affine=self._affine),
                ShuffleNetBlock(inp, oup, mid_channels=mid_channels, ksize=7, stride=stride, affine=self._affine),
                ShuffleXceptionBlock(inp, oup, mid_channels=mid_channels, stride=stride, affine=self._affine)
            ])
            result.append(choice_block)

            if stride == 2:
                self._feature_map_size //= 2
        return result

    def forward(self, x):
        bs = x.size(0)
        x = self.first_conv(x)
        x = self.features(x)
        x = self.conv_last(x)
        x = self.globalpool(x)

        x = self.dropout(x)
        x = x.contiguous().view(bs, -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # FIXME this won't work in base engine
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    torch.nn.init.normal_(m.weight, 0, 0.01)
                else:
                    torch.nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    torch.nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0001)
                torch.nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0001)
                torch.nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)


class LatencyFilter:
    def __init__(self, threshold, predictor, predictor_version=None, reverse=False):
        """
        Filter the models according to predcted latency.

        Parameters
        ----------
        threshold: `float`
            the threshold of latency
        config, hardware:
            determine the targeted device
        reverse: `bool`
            if reverse is `False`, then the model returns `True` when `latency < threshold`,
            else otherwisse
        """
        self.predictors = load_latency_predictor(predictor, predictor_version)
        self.threshold = threshold

    def __call__(self, ir_model):
        latency = self.predictors.predict(ir_model, 'nni-ir')
        return latency < self.threshold


@click.command()
@click.option('--port', default=8081, help='On which port the experiment is run.')
def _main(port):
    base_model = ShuffleNetV2(32)
    base_predictor = 'cortexA76cpu_tflite21'
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

    trainer = pl.Classification(train_dataloader=pl.DataLoader(train_dataset, batch_size=64),
                                val_dataloaders=pl.DataLoader(test_dataset, batch_size=64),
                                max_epochs=2, gpus=1)

    simple_strategy = strategy.Random(model_filter=LatencyFilter(threshold=100, predictor=base_predictor))

    exp = RetiariiExperiment(base_model, trainer, strategy=simple_strategy)

    exp_config = RetiariiExeConfig('local')
    exp_config.trial_concurrency = 2
    exp_config.max_trial_number = 2
    exp_config.trial_gpu_number = 1
    exp_config.training_service.use_active_gpu = False
    exp_config.execution_engine = 'base'
    exp_config.dummy_input = [1, 3, 32, 32]

    exp.run(exp_config, port)

    print('Exported models:')
    for model in exp.export_top_models(formatter='dict'):
        print(model)


if __name__ == '__main__':
    _main()
