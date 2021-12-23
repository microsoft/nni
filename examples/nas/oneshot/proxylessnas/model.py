import torch
import nni.retiarii.nn.pytorch as nn
import math

import ops
import putils
from nni.retiarii.nn.pytorch import LayerChoice

class SearchMobileNet(nn.Module):
    def __init__(self,
                 width_stages=[24,40,80,96,192,320],
                 n_cell_stages=[4,4,4,4,4,1],
                 stride_stages=[2,2,2,1,2,1],
                 width_mult=1, n_classes=1000,
                 dropout_rate=0, bn_param=(0.1, 1e-3)):
        """
        Parameters
        ----------
        width_stages: str
            width (output channels) of each cell stage in the block
        n_cell_stages: str
            number of cells in each cell stage
        stride_strages: str
            stride of each cell stage in the block
        width_mult : int
            the scale factor of width
        """
        super(SearchMobileNet, self).__init__()

        input_channel = putils.make_divisible(32 * width_mult, 8)
        first_cell_width = putils.make_divisible(16 * width_mult, 8)
        for i in range(len(width_stages)):
            width_stages[i] = putils.make_divisible(width_stages[i] * width_mult, 8)
        # first conv
        first_conv = ops.ConvLayer(3, input_channel, kernel_size=3, stride=2, use_bn=True, act_func='relu6', ops_order='weight_bn_act')
        # first block
        first_block_conv = ops.OPS['3x3_MBConv1'](input_channel, first_cell_width, 1)
        first_block = first_block_conv

        input_channel = first_cell_width

        blocks = [first_block]

        stage_cnt = 0
        for width, n_cell, s in zip(width_stages, n_cell_stages, stride_stages):
            for i in range(n_cell):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                op_candidates = [ops.OPS['3x3_MBConv3'](input_channel, width, stride),
                                 ops.OPS['3x3_MBConv6'](input_channel, width, stride),
                                 ops.OPS['5x5_MBConv3'](input_channel, width, stride),
                                 ops.OPS['5x5_MBConv6'](input_channel, width, stride),
                                 ops.OPS['7x7_MBConv3'](input_channel, width, stride),
                                 ops.OPS['7x7_MBConv6'](input_channel, width, stride)]
                if stride == 1 and input_channel == width:
                    # if it is not the first one
                    op_candidates += [ops.OPS['Zero'](input_channel, width, stride)]
                    conv_op = LayerChoice(op_candidates, label="s{}_c{}".format(stage_cnt, i))
                else:
                    conv_op = LayerChoice(op_candidates, label="s{}_c{}".format(stage_cnt, i))
                # shortcut
                if stride == 1 and input_channel == width:
                    # if not first cell
                    shortcut = ops.IdentityLayer(input_channel, input_channel)
                else:
                    shortcut = None
                inverted_residual_block = ops.MobileInvertedResidualBlock(conv_op, shortcut, op_candidates)
                blocks.append(inverted_residual_block)
                input_channel = width
            stage_cnt += 1

        # feature mix layer
        last_channel = putils.make_devisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        feature_mix_layer = ops.ConvLayer(input_channel, last_channel, kernel_size=1, use_bn=True, act_func='relu6', ops_order='weight_bn_act', )
        classifier = ops.LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)

        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.feature_mix_layer = feature_mix_layer
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = classifier

        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.feature_mix_layer(x)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def set_bn_param(self, momentum, eps):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.momentum = momentum
                m.eps = eps
        return

    def init_model(self, model_init='he_fout', init_div_groups=False):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if model_init == 'he_fout':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif model_init == 'he_fin':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                else:
                    raise NotImplementedError
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.zero_()
