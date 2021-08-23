# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F

from nni.nas.pytorch import mutables
from ops import FactorizedReduce, StdConv, SepConvBN, Pool


class AuxiliaryHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.pooling = nn.Sequential(
            nn.ReLU(),
            nn.AvgPool2d(5, 3, 2)
        )
        self.proj = nn.Sequential(
            StdConv(in_channels, 128),
            StdConv(128, 768)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(768, 10, bias=False)

    def forward(self, x):
        bs = x.size(0)
        x = self.pooling(x)
        x = self.proj(x)
        x = self.avg_pool(x).view(bs, -1)
        x = self.fc(x)
        return x


class Cell(nn.Module):
    def __init__(self, cell_name, prev_labels, channels):
        super().__init__()
        self.input_choice = mutables.InputChoice(choose_from=prev_labels, n_chosen=1, return_mask=True,
                                                 key=cell_name + "_input")
        self.op_choice = mutables.LayerChoice([
            SepConvBN(channels, channels, 3, 1),
            SepConvBN(channels, channels, 5, 2),
            Pool("avg", 3, 1, 1),
            Pool("max", 3, 1, 1),
            nn.Identity()
        ], key=cell_name + "_op")

    def forward(self, prev_layers):
        from nni.retiarii.oneshot.pytorch.random import PathSamplingInputChoice
        out = self.input_choice(prev_layers)
        if isinstance(self.input_choice, PathSamplingInputChoice):
            # Retiarii pattern
            return out, self.input_choice.mask
        else:
            chosen_input, chosen_mask = out
            cell_out = self.op_choice(chosen_input)
            return cell_out, chosen_mask


class Node(mutables.MutableScope):
    def __init__(self, node_name, prev_node_names, channels):
        super().__init__(node_name)
        self.cell_x = Cell(node_name + "_x", prev_node_names, channels)
        self.cell_y = Cell(node_name + "_y", prev_node_names, channels)

    def forward(self, prev_layers):
        out_x, mask_x = self.cell_x(prev_layers)
        out_y, mask_y = self.cell_y(prev_layers)
        return out_x + out_y, mask_x | mask_y


class Calibration(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.process = None
        if in_channels != out_channels:
            self.process = StdConv(in_channels, out_channels)

    def forward(self, x):
        if self.process is None:
            return x
        return self.process(x)


class ReductionLayer(nn.Module):
    def __init__(self, in_channels_pp, in_channels_p, out_channels):
        super().__init__()
        self.reduce0 = FactorizedReduce(in_channels_pp, out_channels, affine=False)
        self.reduce1 = FactorizedReduce(in_channels_p, out_channels, affine=False)

    def forward(self, pprev, prev):
        return self.reduce0(pprev), self.reduce1(prev)


class ENASLayer(nn.Module):
    def __init__(self, num_nodes, in_channels_pp, in_channels_p, out_channels, reduction):
        super().__init__()
        self.preproc0 = Calibration(in_channels_pp, out_channels)
        self.preproc1 = Calibration(in_channels_p, out_channels)

        self.num_nodes = num_nodes
        name_prefix = "reduce" if reduction else "normal"
        self.nodes = nn.ModuleList()
        node_labels = [mutables.InputChoice.NO_KEY, mutables.InputChoice.NO_KEY]
        for i in range(num_nodes):
            node_labels.append("{}_node_{}".format(name_prefix, i))
            self.nodes.append(Node(node_labels[-1], node_labels[:-1], out_channels))
        self.final_conv_w = nn.Parameter(torch.zeros(out_channels, self.num_nodes + 2, out_channels, 1, 1), requires_grad=True)
        self.bn = nn.BatchNorm2d(out_channels, affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.final_conv_w)

    def forward(self, pprev, prev):
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


class MicroNetwork(nn.Module):
    def __init__(self, num_layers=2, num_nodes=5, out_channels=24, in_channels=3, num_classes=10,
                 dropout_rate=0.0, use_aux_heads=False):
        super().__init__()
        self.num_layers = num_layers
        self.use_aux_heads = use_aux_heads

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
                self.layers.append(ReductionLayer(c_pp, c_p, c_cur))
                c_pp = c_p = c_cur
            self.layers.append(ENASLayer(num_nodes, c_pp, c_p, c_cur, reduction))
            if self.use_aux_heads and layer_id == pool_layers[-1] + 1:
                self.layers.append(AuxiliaryHead(c_cur, num_classes))
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
        aux_logits = None

        for layer in self.layers:
            if isinstance(layer, AuxiliaryHead):
                if self.training:
                    aux_logits = layer(cur)
            else:
                prev, cur = layer(prev, cur)

        cur = self.gap(F.relu(cur)).view(bs, -1)
        cur = self.dropout(cur)
        logits = self.dense(cur)

        if aux_logits is not None:
            return logits, aux_logits
        return logits
