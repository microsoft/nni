# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F

from nni.nas.pytorch.mutables import LayerChoice, InputChoice, MutableScope


class Cell(MutableScope):
    def __init__(self, cell_name, prev_labels, channels):
        super().__init__(cell_name)
        self.input_choice = InputChoice(choose_from=prev_labels, n_chosen=1, return_mask=True,
                                        key=cell_name + "_input")
        self.op_choice = LayerChoice([
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.Conv2d(channels, channels, 5, padding=2),
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Identity()
        ], key=cell_name + "_op")

    def forward(self, prev_layers):
        chosen_input, chosen_mask = self.input_choice(prev_layers)
        cell_out = self.op_choice(chosen_input)
        return cell_out, chosen_mask


class Node(MutableScope):
    def __init__(self, node_name, prev_node_names, channels):
        super().__init__(node_name)
        self.cell_x = Cell(node_name + "_x", prev_node_names, channels)
        self.cell_y = Cell(node_name + "_y", prev_node_names, channels)

    def forward(self, prev_layers):
        out_x, mask_x = self.cell_x(prev_layers)
        out_y, mask_y = self.cell_y(prev_layers)
        return out_x + out_y, mask_x | mask_y


class Layer(nn.Module):
    def __init__(self, num_nodes, channels):
        super().__init__()
        self.num_nodes = num_nodes
        self.nodes = nn.ModuleList()
        node_labels = [InputChoice.NO_KEY, InputChoice.NO_KEY]
        for i in range(num_nodes):
            node_labels.append("node_{}".format(i))
            self.nodes.append(Node(node_labels[-1], node_labels[:-1], channels))
        self.final_conv_w = nn.Parameter(torch.zeros(channels, self.num_nodes + 2, channels, 1, 1),
                                         requires_grad=True)
        self.bn = nn.BatchNorm2d(channels, affine=False)

    def forward(self, pprev, prev):
        prev_nodes_out = [pprev, prev]
        nodes_used_mask = torch.zeros(self.num_nodes + 2, dtype=torch.bool, device=prev.device)
        for i in range(self.num_nodes):
            node_out, mask = self.nodes[i](prev_nodes_out)
            nodes_used_mask[:mask.size(0)] |= mask.to(prev.device)
            # NOTE: which device should we put mask on?
            prev_nodes_out.append(node_out)

        unused_nodes = torch.cat([out for used, out in zip(nodes_used_mask, prev_nodes_out) if not used], 1)
        unused_nodes = F.relu(unused_nodes)
        conv_weight = self.final_conv_w[:, ~nodes_used_mask, :, :, :]
        conv_weight = conv_weight.view(conv_weight.size(0), -1, 1, 1)
        out = F.conv2d(unused_nodes, conv_weight)
        return prev, self.bn(out)


class SpaceWithMutableScope(nn.Module):
    def __init__(self, test_case, num_layers=4, num_nodes=5, channels=16, in_channels=3, num_classes=10):
        super().__init__()
        self.test_case = test_case
        self.num_layers = num_layers

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers + 2):
            self.layers.append(Layer(num_nodes, channels))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dense = nn.Linear(channels, num_classes)

    def forward(self, x):
        prev = cur = self.stem(x)
        for layer in self.layers:
            prev, cur = layer(prev, cur)

        cur = self.gap(F.relu(cur)).view(x.size(0), -1)
        return self.dense(cur)
