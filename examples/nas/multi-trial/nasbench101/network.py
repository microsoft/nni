import torch.nn as nn

from .base_ops import Conv3x3BnRelu, Conv1x1BnRelu, ConvBnRelu


class NasBench101(nn.Module):
    def __init__(self,
                 stem_out_channels: int = 128,
                 num_stacks: int = 3,
                 num_modules_per_stack: int = 3,
                 max_num_vertices: int = 7,
                 max_num_edges: int = 9,
                 num_labels: int = 10,
                 bn_eps: float = 1e-5,
                 bn_momentum: float = 0.003):
        super().__init__()

        # initial stem convolution
        self.stem_conv = Conv3x3BnRelu(3, stem_out_channels)

        layers = []
        in_channels = out_channels = stem_out_channels
        for stack_num in range(num_stacks):
            if stack_num > 0:
                downsample = nn.MaxPool2d(kernel_size=2, stride=2)
                layers.append(downsample)
                out_channels *= 2
            for _ in range(num_modules_per_stack):
                cell = Cell(max_num_vertices, max_num_edges, in_channels, out_channels)
                layers.append(cell)
                in_channels = out_channels

        self.features = nn.ModuleList(layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(out_channels, num_labels)

        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eps = bn_eps
                module.momentum = bn_momentum

    def forward(self, x):
        bs = x.size(0)
        out = self.stem_conv(x)
        for layer in self.features:
            out = layer(out)
        out = self.gap(out).view(bs, -1)
        out = self.classifier(out)
        return out

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eps = self.config.bn_eps
                module.momentum = self.config.bn_momentum

    def validate(self) -> bool:
        return self.features[0].validate()