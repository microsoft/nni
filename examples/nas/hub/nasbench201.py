# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json

import nni
import numpy as np
import torch

from nni.retiarii import strategy, fixed_arch
from nni.retiarii.evaluator.pytorch import Lightning, ClassificationModule, Trainer
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
from nni.retiarii.hub.pytorch import NasBench201
from torch.utils.data import DataLoader, SubsetRandomSampler
from typing_extensions import Literal

from darts import get_cifar10_dataset


@nni.trace
class NasBench201TrainingModule(ClassificationModule):
    """Adjust momentum, nesterov in SGD optimizer, and add a LR scheduler."""
    model: NasBench201

    def __init__(self,
                 learning_rate: float = 0.1,
                 weight_decay: float = 5e-4,
                 max_epochs: int = 200):
        self.max_epochs = max_epochs
        super().__init__(learning_rate=learning_rate, weight_decay=weight_decay, export_onnx=False)

    def configure_optimizers(self):
        """Customized optimizer with momentum, as well as a scheduler."""
        optimizer = torch.optim.SGD(
            self.parameters(),
            momentum=0.9,
            lr=self.hparams.learning_rate,  # type: ignore
            weight_decay=self.hparams.weight_decay,  # type: ignore
            nesterov=True,
        )
        return {
            'optimizer': optimizer,
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.max_epochs, eta_min=0)
        }


def search(batch_size: int = 256, algo: Literal['ENAS', 'DARTS', 'Gumbel', 'Proxyless'] = 'ENAS', **kwargs):
    model_space = NasBench201()

    train_data = get_cifar10_dataset()
    num_samples = len(train_data)
    indices = np.random.permutation(num_samples)
    split = num_samples // 2

    train_loader = DataLoader(
        train_data, batch_size=batch_size,
        sampler=SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=6
    )

    valid_loader = DataLoader(
        train_data, batch_size=batch_size,
        sampler=SubsetRandomSampler(indices[split:]),
        pin_memory=True, num_workers=6
    )

    evaluator = Lightning(
        NasBench201TrainingModule(),
        Trainer(gpus=1, max_epochs=200),
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader
    )

    strategy_ = getattr(strategy, algo)()

    config = RetiariiExeConfig(execution_engine='oneshot')
    experiment = RetiariiExperiment(model_space, evaluator=evaluator, strategy=strategy_)
    experiment.run(config)

    return experiment.export_top_models()[0]


def train(arch, batch_size: int = 256, **kwargs):
    with fixed_arch(arch):
        model = NasBench201()

    train_data = get_cifar10_dataset(cutout=True)
    valid_data = get_cifar10_dataset(train=False)

    evaluator = Lightning(
        NasBench201TrainingModule(),
        Trainer(gpus=1, max_epochs=200),
        train_dataloaders=DataLoader(train_data, batch_size=batch_size, pin_memory=True, num_workers=6),
        val_dataloaders=DataLoader(valid_data, batch_size=batch_size, pin_memory=True, num_workers=6)
    )

    evaluator.fit(model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['search', 'train', 'search_train'], default='search_train')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--arch', type=str)
    parser.add_argument('--weight_file', type=str)

    parsed_args = parser.parse_args()
    config = {k: v for k, v in vars(parsed_args).items() if v is not None}
    if 'arch' in config:
        config['arch'] = json.loads(config['arch'])

    if 'search' in config['mode']:
        config['arch'] = search(**config)
        print('Searched config', config['arch'])
    if 'train' in config['mode']:
        train(**config)


if __name__ == '__main__':
    main()
