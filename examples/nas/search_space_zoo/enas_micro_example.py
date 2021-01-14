import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

from argparse import ArgumentParser
from torchvision import transforms
from torchvision.datasets import CIFAR10

from nni.algorithms.nas.pytorch import enas
from utils import accuracy, reward_accuracy
from nni.nas.pytorch.callbacks import (ArchitectureCheckpoint,
                                       LRSchedulerCallback)

from nni.nas.pytorch.search_space_zoo import ENASMicroLayer

logger = logging.getLogger('nni')


def get_dataset(cls):
    MEAN = [0.49139968, 0.48215827, 0.44653124]
    STD = [0.24703233, 0.24348505, 0.26158768]
    transf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]

    train_transform = transforms.Compose(transf + normalize)
    valid_transform = transforms.Compose(normalize)

    if cls == "cifar10":
        dataset_train = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
        dataset_valid = CIFAR10(root="./data", train=False, download=True, transform=valid_transform)
    else:
        raise NotImplementedError
    return dataset_train, dataset_valid


class MicroNetwork(nn.Module):
    def __init__(self, num_layers=2, num_nodes=5, out_channels=24, in_channels=3, num_classes=10,
                 dropout_rate=0.0):
        super().__init__()
        self.num_layers = num_layers

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 3, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels * 3)
        )

        pool_distance = self.num_layers // 3
        pool_layers = [pool_distance, 2 * pool_distance + 1]
        self.dropout = nn.Dropout(dropout_rate)

        self.layers = nn.ModuleList()
        c_pp = c_p = out_channels * 3
        c_cur = out_channels
        for layer_id in range(self.num_layers + 2):
            reduction = False
            if layer_id in pool_layers:
                c_cur, reduction = c_p * 2, True
            self.layers.append(ENASMicroLayer(num_nodes, c_pp, c_p, c_cur, reduction))
            if reduction:
                c_pp = c_p = c_cur
            c_pp, c_p = c_p, c_cur

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dense = nn.Linear(c_cur, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        bs = x.size(0)
        prev = cur = self.stem(x)
        # aux_logits = None

        for layer in self.layers:
            prev, cur = layer(prev, cur)

        cur = self.gap(F.relu(cur)).view(bs, -1)
        cur = self.dropout(cur)
        logits = self.dense(cur)

        # if aux_logits is not None:
        #     return logits, aux_logits
        return logits


if __name__ == "__main__":
    parser = ArgumentParser("enas")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--log-frequency", default=10, type=int)
    parser.add_argument("--epochs", default=None, type=int, help="Number of epochs (default: macro 310, micro 150)")
    parser.add_argument("--visualization", default=False, action="store_true")
    args = parser.parse_args()

    dataset_train, dataset_valid = get_dataset("cifar10")

    model = MicroNetwork(num_layers=6, out_channels=20, num_nodes=5, dropout_rate=0.1)
    num_epochs = args.epochs or 150
    mutator = enas.EnasMutator(model, tanh_constant=1.1, cell_exit_extra_step=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 0.05, momentum=0.9, weight_decay=1.0E-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.001)

    trainer = enas.EnasTrainer(model,
                               loss=criterion,
                               metrics=accuracy,
                               reward_function=reward_accuracy,
                               optimizer=optimizer,
                               callbacks=[LRSchedulerCallback(lr_scheduler), ArchitectureCheckpoint("./checkpoints")],
                               batch_size=args.batch_size,
                               num_epochs=num_epochs,
                               dataset_train=dataset_train,
                               dataset_valid=dataset_valid,
                               log_frequency=args.log_frequency,
                               mutator=mutator)
    if args.visualization:
        trainer.enable_visualization()
    trainer.train()

