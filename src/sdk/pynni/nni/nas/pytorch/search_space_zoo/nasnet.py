from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import nni
from nni.nas.pytorch import mutables
from nni.nas.pytorch.mutables import LayerChoice, InputChoice
from torchvision import datasets, transforms

from nasnet_ops import Pool, Identity, SeparableConv2d, StdConv, FactorizedReduce
import logging

logger = logging.getLogger('nasnet')


OPS = lambda in_ch, out_ch, stride, stem: OrderedDict([
    ('maxpool3x3', Pool('max', in_ch, out_ch, 3, stride, 1)),
    ('avgpool3x3', Pool('avg', in_ch, out_ch, 3, stride, 1)),
    ('identity', Identity(in_ch, out_ch, stride)),
    ('sepconv3x3', SeparableConv2d(in_ch, out_ch, 3, stride, 1, stem)),
    ('sepconv5x5', SeparableConv2d(in_ch, out_ch, 5, stride, 2, stem)),
    ('sepconv7x7', SeparableConv2d(in_ch, out_ch, 7, stride, 3, stem)),
])

from nni.nas.pytorch.classic_nas import get_and_apply_next_architecture
from torchvision import transforms
from torchvision.datasets import CIFAR10
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


class Cell(nn.Module):
    def __init__(self, cell_name, prev_labels, channels, stride, stem):
        super(Cell, self).__init__()
        self.input_choice = InputChoice(choose_from=prev_labels,
                                        n_chosen=1,
                                        return_mask=True,
                                        key=cell_name + '_input')
        self.op_choice = LayerChoice(OPS(channels, channels, stride, stem),
                                     key=cell_name + '_op')

    def forward(self, prev_layers):
        chosen_input, chosen_mask = self.input_choice(prev_layers)
        out = self.op_choice(chosen_input)
        return out, chosen_mask


class Node(mutables.MutableScope):
    def __init__(self, node_name, prev_node_names, channels, stride=1, stem=False):
        super(Node, self).__init__(node_name)
        self.cell_x = Cell(node_name + 'x', prev_node_names, channels, stride, stem)
        self.cell_y = Cell(node_name + 'y', prev_node_names, channels, stride, stem)

    def forward(self, prev_layers):
        out_x, mask_x = self.cell_x(prev_layers)
        out_y, mask_y = self.cell_y(prev_layers)
        return out_x + out_y, mask_x | mask_y


class NasNetCell(nn.Module):
    def __init__(self, num_nodes, in_channels_pp, in_channels_p, out_channels, reduction=False):
        super(NasNetCell, self).__init__()
        self.reduction = reduction
        if self.reduction:
            self.reduce0 = FactorizedReduce(in_channels_pp, out_channels, affine=False)
            self.reduce1 = FactorizedReduce(in_channels_p, out_channels, affine=False)
            in_channels_pp = in_channels_p = out_channels
        self.preproc0 = StdConv(in_channels_pp, out_channels) if in_channels_pp != out_channels else nn.Identity()
        self.preproc1 = StdConv(in_channels_p, out_channels) if in_channels_pp != out_channels else nn.Identity()

        self.num_nodes = num_nodes
        name_prefix = "reduce" if reduction else "normal"
        self.nodes = nn.ModuleList()
        node_labels = [mutables.InputChoice.NO_KEY, mutables.InputChoice.NO_KEY]
        for i in range(num_nodes):
            node_labels.append("{}_node_{}".format(name_prefix, i))
            self.nodes.append(Node(node_labels[-1], node_labels[:-1], out_channels))
        self.final_conv_w = nn.Parameter(torch.zeros(out_channels, self.num_nodes + 2, out_channels, 1, 1),
                                         requires_grad=True)
        self.bn = nn.BatchNorm2d(out_channels, affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.final_conv_w)

    def forward(self, prev, pprev):
        if self.reduction:
            pprev, prev = self.reduce0(pprev), self.reduce1(prev)
        pprev_, prev_ = self.preproc0(pprev), self.preproc1(prev)

        prev_nodes_out = [pprev_, prev_]
        nodes_used_mask = torch.zeros(self.num_nodes + 2, dtype=torch.bool, device=prev.device)
        for i in range(self.num_nodes):
            node_out, mask = self.nodes[i](prev_nodes_out)
            nodes_used_mask[:mask.size(0)] |= mask.to(node_out.device)
            prev_nodes_out.append(node_out)

        unused_nodes = torch.cat([out for used, out in zip(nodes_used_mask, prev_nodes_out) if not used], 1)
        unused_nodes = F.relu(unused_nodes)
        conv_weight = self.final_conv_w[:, ~nodes_used_mask, :, :, :]
        conv_weight = conv_weight.view(conv_weight.size(0), -1, 1, 1)
        out = F.conv2d(unused_nodes, conv_weight)
        return prev, self.bn(out)


class NasNet(nn.Module):
    def __init__(self, num_blocks, num_cells, num_nodes=5, num_classes=10, in_channels=3, out_channels=24,
                 dropout_rate=0.0):
        super(NasNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 3, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels * 3)
        )
        self.dropout = nn.Dropout(dropout_rate)
        c_pp = c_p = out_channels * 3
        c_cur = out_channels
        self.layers = nn.ModuleList()
        for block in range(num_blocks - 1):
            for cell in range(num_cells - 1):
                self.layers.append(NasNetCell(num_nodes, c_pp, c_p, c_cur))
                c_pp, c_p = c_p, c_cur
            self.layers.append(NasNetCell(num_nodes, c_pp, c_p, c_cur, reduction=True))
            c_pp = c_p = c_cur

        for cell in range(num_cells - 1):
            self.layers.append(NasNetCell(num_nodes, c_pp, c_p, c_cur))
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

        for layer in self.layers:
            prev, cur = layer(prev, cur)

        cur = self.gap(F.relu(cur)).view(bs, -1)
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


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 50:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


if __name__ == "__main__":
    model = NasNet(num_blocks=3, num_cells=4, dropout_rate=0.1)
    get_and_apply_next_architecture(model)
    num_epochs = 150
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 0.05, momentum=0.9, weight_decay=1.0E-4)
    device = torch.device("cuda")
    # data_dir = './data'
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(data_dir, train=True, download=True,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=1000, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(data_dir, train=False, transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.1307,), (0.3081,))
    #     ])),
    #     batch_size=1000, shuffle=True)

    train_, test = get_dataset('cifar10')
    data_loader = torch.utils.data.DataLoader(train_,
                                              batch_size=4,
                                              shuffle=True)

    for epoch in range(1, num_epochs + 1):
        train(model, device, data_loader, optimizer, epoch)
        # test_acc = test(args, model, device, test_loader)

