from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import nni
from nni.nas.pytorch import mutables
from nni.nas.pytorch.mutables import LayerChoice, InputChoice
from torchvision import datasets, transforms

from .nasnet_ops import Pool, Identity, SeparableConv2d, StdConv
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
    def __init__(self, cell_name, prev_labels, in_ch, out_ch, stride, stem):
        super(Cell, self).__init__()
        self.input_choice = InputChoice(choose_from=prev_labels,
                                        n_chosen=1,
                                        return_mask=True,
                                        key=cell_name + '_input')
        self.op_choice = LayerChoice(OPS(in_ch, out_ch, stride, stem),
                                     key=cell_name + '_op')

    def forward(self, prev_layers):
        chosen_input, chosen_mask = self.input_choice(prev_layers)
        out = self.op_choice(chosen_input)
        return out, chosen_mask


class Node(mutables.MutableScope):
    def __init__(self, node_name, prev_node_names, in_ch, out_ch, stride, stem):
        super(Node, self).__init__(node_name)
        self.cell_x = Cell(node_name + 'x', prev_node_names, in_ch, out_ch, stride, stem)
        self.cell_y = Cell(node_name + 'y', prev_node_names, in_ch, out_ch, stride, stem)

    def forward(self, prev_layers):
        out_x, mask_x = self.cell_x(prev_layers)
        out_y, mask_y = self.cell_y(prev_layers)
        return out_x + out_y, mask_x | mask_y


class NasNetCell(nn.Module):
    def __init__(self, n_hidden, in_ch, in_ch_prev, out_ch, reduction=False, reduction_p=False,
                 stem=False, channels=None):
        # TODO: hint: num of chosen -> user
        # helper: json -> num
        # hack classic nas mutator -> return json file -> dict: from mutate name to channel num
        super(NasNetCell, self).__init__()
        self.n_hidden = n_hidden
        self.reduction = reduction
        self.reduction_p = reduction_p

        if reduction_p:
            self.p1 = nn.Sequential(
                nn.AvgPool2d(1, stride=2, count_include_pad=False),
                nn.Conv2d(in_ch, out_ch // 2, 1, stride=1, bias=False)
            )
            self.p2 = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.AvgPool2d(1, stride=2, count_include_pad=False),
                nn.Conv2d(in_ch, (out_ch + 1) // 2, 1, stride=1, bias=False)
            )
            self.p_bn = nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.1, affine=True)
        else:
            self.conv_prev_1x1 = StdConv(in_ch_prev, out_ch)
        self.conv_1x1 = StdConv(in_ch, out_ch)
        if channels is not None:
            self.channels = channels
        else:
            self.channels = [out_ch] * (self.n_hidden + 2)
        name_prefix = "reduce" if reduction else "normal"
        self.nodes = nn.ModuleList()
        node_labels = [mutables.InputChoice.NO_KEY, mutables.InputChoice.NO_KEY]
        for i in range(n_hidden):
            node_labels.append("{}_node_{}".format(name_prefix, i))
            self.nodes.append(Node(node_name=node_labels[-1],
                                   prev_node_names=node_labels[:-1],
                                   in_ch=out_ch,  # TODO: change to self.channels
                                   out_ch=out_ch,
                                   stride=2 if reduction and out_ch < 2 else 1,
                                   stem=stem))
        self.final_conv_w = nn.Parameter(torch.zeros(out_ch, self.num_nodes + 2, out_ch, 1, 1),
                                         requires_grad=True)

    def forward(self, x, x_prev):
        if self.reduction_p:
            relu_x = F.relu(x_prev)
            x_p1 = self.p1(relu_x)
            x_p2 = self.p2(relu_x)
            x_prev = self.p_bn(torch.cat([x_p1, x_p2], 1))
        else:
            x_prev = self.conv_prev_1x1(x_prev)
        x = self.conv_1x1(x)
        prev_nodes_out = [x_prev, x]
        nodes_used_mask = torch.zeros(self.n_hidden + 2, dtype=torch.bool, device=x_prev.device)
        for i in range(self.num_nodes):
            node_out, mask = self.nodes[i](prev_nodes_out)
            nodes_used_mask[:mask.size(0)] |= mask.to(node_out.device)
            prev_nodes_out.append(node_out)
        unused_nodes = torch.cat([out for used, out in zip(nodes_used_mask, prev_nodes_out) if not used], 1)
        unused_nodes = F.relu(unused_nodes)
        conv_weight = self.final_conv_w[:, ~nodes_used_mask, :, :, :]
        conv_weight = conv_weight.view(conv_weight.size(0), -1, 1, 1)
        out = F.conv2d(unused_nodes, conv_weight)
        return out


class NasNet(nn.Module):
    def __init__(self, num_stem_features, num_normal_cells, filters, scaling=2, n_hidden=5, num_classes=1000):
        super(NasNet, self).__init__()
        self.num_normal_cells = num_normal_cells
        self.num_classes = num_classes
        self.conv0 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(3, num_stem_features, kernel_size=3, stride=2, bias=False)),
            ('bn', nn.BatchNorm2d(num_stem_features, eps=0.001, momentum=0.1, affine=True))
        ]))
        filters_p = filters_pp = num_stem_features
        self.conv0 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(3, num_stem_features, kernel_size=3, stride=2, bias=False)),
            ('bn', nn.BatchNorm2d(num_stem_features, eps=0.001, momentum=0.1, affine=True))
        ]))
        self.conv1 = nn.Linear(num_stem_features, filters_pp)
        self.conv2 = nn.Linear(num_stem_features, filters_p)
        #
        # expand_normal = expand_normal
        # expand_reduce = expand_reduce

        self.layers = nn.ModuleList()
        for block in range(3):
            for _ in range(num_normal_cells - 1):
                self.layers.append(NasNetCell(n_hidden, filters_p, filters_pp, filters))
                filters_pp, filters_p = filters_p, filters
            filters *= scaling
            if block == 0 or block == 1:
                self.layers.append(NasNetCell(n_hidden, filters_p, filters_pp, filters, reduction=True))
                filters_p = filters

        self.linear = nn.Linear(filters_p, self.num_classes)

    def forward(self, x):
        x_conv0 = self.conv0(x)
        x_stem_0 = self.cell_stem_0(x_conv0)
        x_stem_1 = self.cell_stem_1(x_conv0)

        prev_x, x = x_stem_0, x_stem_1
        for block in range(3):
            for layer in self.layers:
                cur = layer(x, prev_x)
                prev_x, x = x, cur
        
        output = self.logits(x)
        return output

    def logits(self, features):
        x = F.relu(features, inplace=False)
        x = F.avg_pool2d(x, kernel_size=x.size(2)).view(x.size(0), -1)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear(x)
        return x


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
    model = NasNet(num_stem_features=32, num_normal_cells=4, filters=44, scaling=2)
    get_and_apply_next_architecture(model)
    num_epochs = 150
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 0.05, momentum=0.9, weight_decay=1.0E-4)
    device = torch.device("cuda")
    data_dir = './data'
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=1000, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1000, shuffle=True)

    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        # test_acc = test(args, model, device, test_loader)

