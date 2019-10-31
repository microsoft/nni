from argparse import ArgumentParser

import datasets
import image_ops as ops
import nni.nas.pytorch as nas
import torch
import torch.nn as nn
from nni.nas.pytorch.darts import DartsTrainer


class SearchCell(nn.Module):
    """
    Cell for search.
    """

    def __init__(self, n_nodes, channels_pp, channels_p, channels, reduction_p, reduction):
        """
        Initialization a search cell.

        Parameters
        ----------
        n_nodes: int
            Number of nodes in current DAG.
        channels_pp: int
            Number of output channels from previous previous cell.
        channels_p: int
            Number of output channels from previous cell.
        channels: int
            Number of channels that will be used in the current DAG.
        reduction_p: bool
            Flag for whether the previous cell is reduction cell or not.
        reduction: bool
            Flag for whether the current cell is reduction cell or not.
        """
        super().__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        if reduction_p:
            self.preproc0 = ops.FactorizedReduce(channels_pp, channels, affine=False)
        else:
            self.preproc0 = ops.StdConv(channels_pp, channels, 1, 1, 0, affine=False)
        self.preproc1 = ops.StdConv(channels_p, channels, 1, 1, 0, affine=False)

        # generate dag
        self.mutable_ops = nn.ModuleList()
        for depth in range(self.n_nodes):
            self.mutable_ops.append(nn.ModuleList())
            for i in range(2 + depth):  # include 2 input nodes
                # reduction should be used only for input node
                stride = 2 if reduction and i < 2 else 1
                op = nas.mutables.LayerChoice([ops.PoolBN('max', channels, 3, stride, 1, affine=False),
                                               ops.PoolBN('avg', channels, 3, stride, 1, affine=False),
                                               ops.Identity() if stride == 1 else
                                               ops.FactorizedReduce(channels, channels, affine=False),
                                               ops.SepConv(channels, channels, 3, stride, 1, affine=False),
                                               ops.SepConv(channels, channels, 5, stride, 2, affine=False),
                                               ops.DilConv(channels, channels, 3, stride, 2, 2, affine=False),
                                               ops.DilConv(channels, channels, 5, stride, 4, 2, affine=False),
                                               ops.Zero(stride)],
                                              key="r{}_d{}_i{}".format(reduction, depth, i))
                self.mutable_ops[depth].append(op)

    def forward(self, s0, s1):
        # s0, s1 are the outputs of previous previous cell and previous cell, respectively.
        tensors = [self.preproc0(s0), self.preproc1(s1)]
        for ops in self.mutable_ops:
            assert len(ops) == len(tensors)
            cur_tensor = sum(op(tensor) for op, tensor in zip(ops, tensors))
            tensors.append(cur_tensor)

        output = torch.cat(tensors[2:], dim=1)
        return output


class SearchCNN(nn.Module):
    """
    Search CNN model
    """

    def __init__(self, in_channels, channels, n_classes, n_layers, n_nodes=4, stem_multiplier=3):
        """
        Initializing a search channelsNN.

        Parameters
        ----------
        in_channels: int
            Number of channels in images.
        channels: int
            Number of channels used in the network.
        n_classes: int
            Number of classes.
        n_layers: int
            Number of cells in the whole network.
        n_nodes: int
            Number of nodes in a cell.
        stem_multiplier: int
            Multiplier of channels in STEM.
        """
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.n_classes = n_classes
        self.n_layers = n_layers

        c_cur = stem_multiplier * self.channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c_cur)
        )

        # for the first cell, stem is used for both s0 and s1
        # [!] channels_pp and channels_p is output channel size, but c_cur is input channel size.
        channels_pp, channels_p, c_cur = c_cur, c_cur, channels

        self.cells = nn.ModuleList()
        reduction_p, reduction = False, False
        for i in range(n_layers):
            reduction_p, reduction = reduction, False
            # Reduce featuremap size and double channels in 1/3 and 2/3 layer.
            if i in [n_layers // 3, 2 * n_layers // 3]:
                c_cur *= 2
                reduction = True

            cell = SearchCell(n_nodes, channels_pp, channels_p, c_cur, reduction_p, reduction)
            self.cells.append(cell)
            c_cur_out = c_cur * n_nodes
            channels_pp, channels_p = channels_p, c_cur_out

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(channels_p, n_classes)

    def forward(self, x):
        s0 = s1 = self.stem(x)

        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)

        out = self.gap(s1)
        out = out.view(out.size(0), -1)  # flatten
        logits = self.linear(out)
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


if __name__ == "__main__":
    parser = ArgumentParser("darts")
    parser.add_argument("--layers", default=4, type=int)
    parser.add_argument("--nodes", default=2, type=int)
    parser.add_argument("--batch-size", default=3, type=int)
    parser.add_argument("--log-frequency", default=1, type=int)
    args = parser.parse_args()

    dataset_train, dataset_valid = datasets.get_dataset("cifar10")

    model = SearchCNN(3, 16, 10, args.layers, n_nodes=args.nodes)
    criterion = nn.CrossEntropyLoss()

    optim = torch.optim.SGD(model.parameters(), 0.025, momentum=0.9, weight_decay=3.0E-4)
    n_epochs = 50
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, n_epochs, eta_min=0.001)

    trainer = DartsTrainer(model,
                           loss=criterion,
                           metrics=lambda output, target: accuracy(output, target, topk=(1,)),
                           model_optim=optim,
                           lr_scheduler=lr_scheduler,
                           num_epochs=50,
                           dataset_train=dataset_train,
                           dataset_valid=dataset_valid,
                           batch_size=args.batch_size,
                           log_frequency=args.log_frequency)
    trainer.train()
    trainer.finalize()

# augment step
# ...
