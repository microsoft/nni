import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

from argparse import ArgumentParser
from torchvision import transforms
from torchvision.datasets import CIFAR10

from nni.nas.pytorch import mutables
from nni.nas.pytorch import enas
from utils import accuracy, reward_accuracy
from nni.nas.pytorch.callbacks import (ArchitectureCheckpoint,
                                       LRSchedulerCallback)
from nni.nas.pytorch.search_space_zoo import ENASMacroLayer

logger = logging.getLogger('nni')


class GeneralNetwork(nn.Module):
    def __init__(self, num_layers=12, out_filters=24, in_channels=3, num_classes=10,
                 dropout_rate=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.out_filters = out_filters

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_filters, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_filters)
        )

        pool_distance = self.num_layers // 3
        self.pool_layers_idx = [pool_distance - 1, 2 * pool_distance - 1]
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)

        self.layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        labels = []
        for layer_id in range(self.num_layers):
            labels.append("layer_{}".format(layer_id))
            if layer_id in self.pool_layers_idx:
                self.pool_layers.append(FactorizedReduce(self.out_filters, self.out_filters))
            self.layers.append(ENASLayer(labels[-1], labels[:-1], self.out_filters, self.out_filters))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dense = nn.Linear(self.out_filters, self.num_classes)

    def forward(self, x):
        bs = x.size(0)
        cur = self.stem(x)

        layers = [cur]

        for layer_id in range(self.num_layers):
            cur = self.layers[layer_id](layers)
            layers.append(cur)
            if layer_id in self.pool_layers_idx:
                for i, layer in enumerate(layers):
                    layers[i] = self.pool_layers[self.pool_layers_idx.index(layer_id)](layer)
                cur = layers[-1]

        cur = self.gap(cur).view(bs, -1)
        cur = self.dropout(cur)
        logits = self.dense(cur)
        return logits


if __name__ == "__main__":
    parser = ArgumentParser("enas")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--log-frequency", default=10, type=int)
    # parser.add_argument("--search-for", choices=["macro", "micro"], default="macro")
    parser.add_argument("--epochs", default=None, type=int, help="Number of epochs (default: macro 310, micro 150)")
    parser.add_argument("--visualization", default=False, action="store_true")
    args = parser.parse_args()

    dataset_train, dataset_valid = datasets.get_dataset("cifar10")
    model = GeneralNetwork()
    num_epochs = args.epochs or 310
    mutator = None

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
