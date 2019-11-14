
import torch.nn as nn

from .cnn_cell import CnnCell


class CnnNetwork(nn.Module):
    """
    Search CNN model
    """

    def __init__(self, in_channels, channels, n_classes, n_layers, n_nodes=4, stem_multiplier=3, cell_type=CnnCell):
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

            cell = cell_type(n_nodes, channels_pp, channels_p, c_cur, reduction_p, reduction)
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
