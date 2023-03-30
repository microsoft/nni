# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Union, cast

import torch
import torch.nn as nn
import nni
from nni.mutable import MutableExpression
from nni.nas.nn.pytorch import LayerChoice, ModelSpace, MutableConv2d, MutableBatchNorm2d

from .utils.pretrained import load_pretrained_weight


class ShuffleNetBlock(nn.Module):
    """
    Describe the basic building block of shuffle net, as described in
    `ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices <https://arxiv.org/pdf/1707.01083.pdf>`__.

    When stride = 1, the block expects an input with ``2 * input channels``. Otherwise input channels.
    """

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Union[int, MutableExpression[int]], *,
                 kernel_size: int, stride: int, sequence: str = "pdp", affine: bool = True):
        super().__init__()
        assert stride in [1, 2]
        assert kernel_size in [3, 5, 7]
        self.channels = in_channels // 2 if stride == 1 else in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = kernel_size // 2
        self.oup_main = out_channels - self.channels
        self.affine = affine
        assert self.oup_main > 0

        self.branch_main = nn.Sequential(*self._decode_point_depth_conv(sequence))

        if stride == 2:
            self.branch_proj = nn.Sequential(
                # dw
                MutableConv2d(self.channels, self.channels, kernel_size, stride, self.pad,
                              groups=self.channels, bias=False),
                MutableBatchNorm2d(self.channels, affine=affine),
                # pw-linear
                MutableConv2d(self.channels, self.channels, 1, 1, 0, bias=False),
                MutableBatchNorm2d(self.channels, affine=affine),
                nn.ReLU(inplace=True)
            )
        else:
            # empty block to be compatible with torchscript
            self.branch_proj = nn.Sequential()

    def forward(self, x):
        if self.stride == 2:
            x_proj, x = self.branch_proj(x), x
        else:
            x_proj, x = self._channel_shuffle(x)
        return torch.cat((x_proj, self.branch_main(x)), 1)

    def _decode_point_depth_conv(self, sequence):
        result = []
        first_depth = first_point = True
        pc: int = self.channels
        c: int = self.channels
        for i, token in enumerate(sequence):
            # compute output channels of this conv
            if i + 1 == len(sequence):
                assert token == "p", "Last conv must be point-wise conv."
                c = self.oup_main
            elif token == "p" and first_point:
                c = cast(int, self.mid_channels)
            if token == "d":
                # depth-wise conv
                if isinstance(pc, int) and isinstance(c, int):
                    # check can only be done for static channels
                    assert pc == c, "Depth-wise conv must not change channels."
                result.append(MutableConv2d(pc, c, self.kernel_size, self.stride if first_depth else 1, self.pad,
                                            groups=c, bias=False))
                result.append(MutableBatchNorm2d(c, affine=self.affine))
                first_depth = False
            elif token == "p":
                # point-wise conv
                result.append(MutableConv2d(pc, c, 1, 1, 0, bias=False))
                result.append(MutableBatchNorm2d(c, affine=self.affine))
                result.append(nn.ReLU(inplace=True))
                first_point = False
            else:
                raise ValueError("Conv sequence must be d and p.")
            pc = c
        return result

    def _channel_shuffle(self, x):
        bs, num_channels, height, width = x.size()
        # NOTE: this line is commented for torchscript
        # assert (num_channels % 4 == 0)
        x = x.reshape(bs * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]


class ShuffleXceptionBlock(ShuffleNetBlock):
    """
    The ``choice_x`` version of shuffle net block, described in
    `Single Path One-shot <https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610528.pdf>`__.
    """

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Union[int, MutableExpression[int]],
                 *, stride: int, affine: bool = True):
        super().__init__(in_channels, out_channels, mid_channels,
                         kernel_size=3, stride=stride, sequence="dpdpdp", affine=affine)


class ShuffleNetSpace(ModelSpace):
    """
    The search space proposed in `Single Path One-shot <https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610528.pdf>`__.

    The basic building block design is inspired by a state-of-the-art manually-designed network --
    `ShuffleNetV2 <https://openaccess.thecvf.com/content_ECCV_2018/html/Ningning_Light-weight_CNN_Architecture_ECCV_2018_paper.html>`__.
    There are 20 choice blocks in total. Each choice block has 4 candidates, namely ``choice 3``, ``choice 5``,
    ``choice_7`` and ``choice_x`` respectively. They differ in kernel sizes and the number of depthwise convolutions.
    The size of the search space is :math:`4^{20}`.

    Parameters
    ----------
    num_labels : int
        Number of classes for the classification head. Default: 1000.
    channel_search : bool
        If true, for each building block, the number of ``mid_channels``
        (output channels of the first 1x1 conv in each building block) varies from 0.2x to 1.6x (quantized to multiple of 0.2).
        Here, "k-x" means k times the number of default channels.
        Otherwise, 1.0x is used by default. Default: false.
    affine : bool
        Apply affine to all batch norm. Default: true.
    """

    def __init__(self,
                 num_labels: int = 1000,
                 channel_search: bool = False,
                 affine: bool = True):
        super().__init__()

        self.num_labels = num_labels
        self.channel_search = channel_search
        self.affine = affine

        # the block number in each stage. 4 stages in total. 20 blocks in total.
        self.stage_repeats = [4, 4, 8, 4]

        # output channels for all stages, including the very first layer and the very last layer
        self.stage_out_channels = [-1, 16, 64, 160, 320, 640, 1024]

        # building first layer
        out_channels = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, out_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        feature_blocks = []

        global_block_idx = 0
        for stage_idx, num_repeat in enumerate(self.stage_repeats):
            for block_idx in range(num_repeat):
                # count global index to give names to choices
                global_block_idx += 1

                # get ready for input and output
                in_channels = out_channels
                out_channels = self.stage_out_channels[stage_idx + 2]
                stride = 2 if block_idx == 0 else 1

                # mid channels can be searched
                base_mid_channels = out_channels // 2
                if self.channel_search:
                    k_choice_list = [int(base_mid_channels * (.2 * k)) for k in range(1, 9)]
                    mid_channels = nni.choice(f'channel_{global_block_idx}', k_choice_list)
                else:
                    mid_channels = int(base_mid_channels)

                mid_channels = cast(Union[int, MutableExpression[int]], mid_channels)

                choice_block = LayerChoice(dict(
                    k3=ShuffleNetBlock(in_channels, out_channels, mid_channels=mid_channels, kernel_size=3, stride=stride, affine=affine),
                    k5=ShuffleNetBlock(in_channels, out_channels, mid_channels=mid_channels, kernel_size=5, stride=stride, affine=affine),
                    k7=ShuffleNetBlock(in_channels, out_channels, mid_channels=mid_channels, kernel_size=7, stride=stride, affine=affine),
                    xcep=ShuffleXceptionBlock(in_channels, out_channels, mid_channels=mid_channels, stride=stride, affine=affine)
                ), label=f'layer_{global_block_idx}')
                feature_blocks.append(choice_block)

        self.features = nn.Sequential(*feature_blocks)

        # final layers
        last_conv_channels = self.stage_out_channels[-1]
        self.conv_last = nn.Sequential(
            nn.Conv2d(out_channels, last_conv_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(last_conv_channels, affine=affine),
            nn.ReLU(inplace=True),
        )
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(last_conv_channels, num_labels, bias=False),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.first_conv(x)
        x = self.features(x)
        x = self.conv_last(x)

        x = self.globalpool(x)

        x = self.dropout(x)
        x = x.contiguous().view(-1, self.stage_out_channels[-1])
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    torch.nn.init.normal_(m.weight, 0, 0.01)  # type: ignore
                else:
                    torch.nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])  # type: ignore
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)  # type: ignore
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    torch.nn.init.constant_(m.weight, 1)  # type: ignore
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0001)  # type: ignore
                if m.running_mean is not None:
                    torch.nn.init.constant_(m.running_mean, 0)  # type: ignore
            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    torch.nn.init.constant_(m.weight, 1)  # type: ignore
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0001)  # type: ignore
                if m.running_mean is not None:
                    torch.nn.init.constant_(m.running_mean, 0)  # type: ignore
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)  # type: ignore
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)  # type: ignore

    @classmethod
    def load_searched_model(
        cls, name: str,
        pretrained: bool = False, download: bool = False, progress: bool = True
    ) -> nn.Module:
        if name == 'spos':
            # NOTE: Need BGR tensor, with no normalization
            # https://github.com/ultmaster/spacehub-conversion/blob/371a4fd6646b4e11eda3f61187f7c9a1d484b1ca/cutils.py#L63
            arch = {
                'layer_1': 'k7',
                'layer_2': 'k5',
                'layer_3': 'k3',
                'layer_4': 'k5',
                'layer_5': 'k7',
                'layer_6': 'k3',
                'layer_7': 'k7',
                'layer_8': 'k3',
                'layer_9': 'k7',
                'layer_10': 'k3',
                'layer_11': 'k7',
                'layer_12': 'xcep',
                'layer_13': 'k3',
                'layer_14': 'k3',
                'layer_15': 'k3',
                'layer_16': 'k3',
                'layer_17': 'xcep',
                'layer_18': 'k7',
                'layer_19': 'xcep',
                'layer_20': 'xcep'
            }

        else:
            raise ValueError(f'Unsupported architecture with name: {name}')

        model_factory = cls.frozen_factory(arch)
        model = model_factory()

        if pretrained:
            weight_file = load_pretrained_weight(name, download=download, progress=progress)
            pretrained_weights = torch.load(weight_file)
            model.load_state_dict(pretrained_weights)

        return model
