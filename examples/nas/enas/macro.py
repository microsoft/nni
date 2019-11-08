from argparse import ArgumentParser
import torch
import torch.nn as nn

import datasets
from ops import FactorizedReduce, ConvBranch, PoolBranch
from nni.nas.pytorch import mutables, enas


class ENASLayer(nn.Module):

    def __init__(self, layer_id, in_filters, out_filters):
        super().__init__()
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.mutable = mutables.LayerChoice([
            ConvBranch(in_filters, out_filters, 3, 1, 1, separable=False),
            ConvBranch(in_filters, out_filters, 3, 1, 1, separable=True),
            ConvBranch(in_filters, out_filters, 5, 1, 2, separable=False),
            ConvBranch(in_filters, out_filters, 5, 1, 2, separable=True),
            PoolBranch('avg', in_filters, out_filters, 3, 1, 1),
            PoolBranch('max', in_filters, out_filters, 3, 1, 1)
        ])
        if layer_id > 0:
            self.skipconnect = mutables.InputChoice(layer_id, n_selected=None, reduction="sum")
        else:
            self.skipconnect = None
        self.batch_norm = nn.BatchNorm2d(out_filters, affine=False)
        self.mutable_scope = mutables.MutableScope("layer_{}".format(layer_id))

    def forward(self, prev_layers):
        with self.mutable_scope:
            out = self.mutable(prev_layers[-1])
            if self.skipconnect is not None:
                connection = self.skipconnect(prev_layers[:-1],
                                              ["layer_{}".format(i) for i in range(len(prev_layers) - 1)])
                if connection is not None:
                    out += connection
            return self.batch_norm(out)


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
        for layer_id in range(self.num_layers):
            if layer_id in self.pool_layers_idx:
                self.pool_layers.append(FactorizedReduce(self.out_filters, self.out_filters))
            self.layers.append(ENASLayer(layer_id, self.out_filters, self.out_filters))

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


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = dict()
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res["acc{}".format(k)] = correct_k.mul_(1.0 / batch_size).item()
    return res


def reward_accuracy(output, target, topk=(1,)):
    batch_size = target.size(0)
    _, predicted = torch.max(output.data, 1)
    return (predicted == target).sum().item() / batch_size


if __name__ == "__main__":
    parser = ArgumentParser("enas")
    parser.add_argument("--batch-size", default=3, type=int)
    parser.add_argument("--log-frequency", default=1, type=int)
    args = parser.parse_args()

    dataset_train, dataset_valid = datasets.get_dataset("cifar10")

    model = GeneralNetwork()
    criterion = nn.CrossEntropyLoss()

    n_epochs = 310
    optim = torch.optim.SGD(model.parameters(), 0.05, momentum=0.9, weight_decay=1.0E-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_epochs, eta_min=0.001)

    trainer = enas.EnasTrainer(model,
                               loss=criterion,
                               metrics=accuracy,
                               reward_function=reward_accuracy,
                               optimizer=optim,
                               lr_scheduler=lr_scheduler,
                               batch_size=args.batch_size,
                               num_epochs=n_epochs,
                               dataset_train=dataset_train,
                               dataset_valid=dataset_valid,
                               log_frequency=args.log_frequency)
    trainer.train()
