# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Reproduction of experiments in `DARTS paper <https://arxiv.org/abs/1806.09055>`__.
"""

import argparse
import json

import numpy as np
import torch
import nni
from nni.retiarii import fixed_arch
from nni.retiarii.evaluator.pytorch import Lightning, ClassificationModule, Trainer
from nni.retiarii.hub.pytorch import DARTS
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
from nni.retiarii.strategy import DARTS as DartsStrategy
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import CIFAR10


@nni.trace
class AuxLossClassificationModule(ClassificationModule):
    """Several customization for the training of DARTS, based on default Classification."""
    model: DARTS

    def __init__(self,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.,
                 auxiliary_loss_weight: float = 0.4,
                 max_epochs: int = 600):
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.max_epochs = max_epochs
        super().__init__(learning_rate=learning_rate, weight_decay=weight_decay, export_onnx=False)

    def configure_optimizers(self):
        """Customized optimizer with momentum, as well as a scheduler."""
        optimizer = torch.optim.SGD(
            self.parameters(),
            momentum=0.9,
            lr=self.hparams.learning_rate,  # type: ignore
            weight_decay=self.hparams.weight_decay  # type: ignore
        )
        return {
            'optimizer': optimizer,
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.max_epochs, eta_min=1e-3)
        }

    def training_step(self, batch, batch_idx):
        """Training step, customized with auxiliary loss."""
        x, y = batch
        if self.auxiliary_loss_weight:
            y_hat, y_aux = self(x)
            loss_main = self.criterion(y_hat, y)
            loss_aux = self.criterion(y_aux, y)
            self.log('train_loss_main', loss_main)
            self.log('train_loss_aux', loss_aux)
            loss = loss_main + self.auxiliary_loss_weight * loss_aux
        else:
            y_hat = self(x)
            loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        for name, metric in self.metrics.items():
            self.log('train_' + name, metric(y_hat, y), prog_bar=True)
        return loss

    def on_train_epoch_start(self):
        """Set drop path probability before every epoch. This has no effect if drop path is not enabled in model."""
        self.model.set_drop_path_prob(self.current_epoch / self.max_epochs)


def cutout(img, length: int = 16):
    h, w = img.size(1), img.size(2)
    mask = np.ones((h, w), np.float32)
    y = np.random.randint(h)
    x = np.random.randint(w)

    y1 = np.clip(y - length // 2, 0, h)
    y2 = np.clip(y + length // 2, 0, h)
    x1 = np.clip(x - length // 2, 0, w)
    x2 = np.clip(x + length // 2, 0, w)

    mask[y1: y2, x1: x2] = 0.
    mask = torch.from_numpy(mask)
    mask = mask.expand_as(img)
    img *= mask
    return img


def get_cifar10_dataset(train: bool = True, cutout: bool = False):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        if cutout:
            transform.transforms.append(cutout)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

    return nni.trace(CIFAR10)(root='./data', train=train, download=True, transform=transform)


def search(batch_size: int = 64, **kwargs):
    model_space = DARTS(16, 8, 'cifar')

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
        AuxLossClassificationModule(0.025, 3e-4, 0., 50),
        Trainer(gpus=1, max_epochs=50),
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader
    )

    # Gradient clip needs to be put here because DARTS strategy doesn't support this configuration from trainer.
    strategy = DartsStrategy(gradient_clip_val=5.)

    config = RetiariiExeConfig(execution_engine='oneshot')
    experiment = RetiariiExperiment(model_space, evaluator=evaluator, strategy=strategy)
    experiment.run(config)

    return experiment.export_top_models()[0]


def train(arch, batch_size: int = 96, **kwargs):
    with fixed_arch(arch):
        model = DARTS(36, 20, 'cifar', auxiliary_loss=True, drop_path_prob=0.2)

    train_data = get_cifar10_dataset(cutout=True)
    valid_data = get_cifar10_dataset(train=False)

    evaluator = Lightning(
        AuxLossClassificationModule(0.025, 3e-4, 0., 600),
        Trainer(gpus=1, gradient_clip_val=5., max_epochs=600),
        train_dataloaders=DataLoader(train_data, batch_size=batch_size, pin_memory=True, num_workers=6),
        val_dataloaders=DataLoader(valid_data, batch_size=batch_size, pin_memory=True, num_workers=6)
    )

    evaluator.fit(model)


def test(arch, weight_file, batch_size: int = 512, **kwargs):
    with fixed_arch(arch):
        model = DARTS(36, 20, 'cifar')
    model.load_state_dict(torch.load(weight_file))

    lightning_module = AuxLossClassificationModule(0.025, 3e-4, 0., 600)
    lightning_module.set_model(model)
    trainer = Trainer(gpus=1)

    valid_data = get_cifar10_dataset(train=False)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, pin_memory=True, num_workers=6)

    trainer.validate(lightning_module, valid_loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['search', 'train', 'test', 'search_train'], default='search_train')
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
    if config['mode'] == 'test':
        test(**config)


if __name__ == '__main__':
    main()
