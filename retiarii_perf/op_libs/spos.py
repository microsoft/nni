import os
import pickle
import re

import torch
import torch.nn as nn

#====================Base model

BATCH_SIZE = 256

class ShuffleNetBlock(nn.Module):
    """
    When stride = 1, the block receives input with 2 * inp channels. Otherwise inp channels.
    """

    def __init__(self, inp, oup, mid_channels, ksize, stride, sequence="pdp"):
        super().__init__()
        assert stride in [1, 2]
        assert ksize in [3, 5, 7]
        self.channels = inp // 2 if stride == 1 else inp
        self.inp = inp
        self.oup = oup
        self.mid_channels = mid_channels
        self.ksize = ksize
        self.stride = stride
        self.pad = ksize // 2
        self.oup_main = oup - self.channels
        self.sequence = sequence
        assert self.oup_main > 0

        self.branch_main = nn.Sequential(*self._decode_point_depth_conv(sequence))

        if stride == 2:
            self.branch_proj = nn.Sequential(
                # dw
                nn.Conv2d(self.channels, self.channels, ksize, stride, self.pad,
                          groups=self.channels, bias=False),
                nn.BatchNorm2d(self.channels, affine=False),
                # pw-linear
                nn.Conv2d(self.channels, self.channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.channels, affine=False),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        if self.stride == 2:
            x_proj, x = self.branch_proj(x), x
        else:
            x_proj, x = self._channel_shuffle(x)
        return torch.cat((x_proj, self.branch_main(x)), 1)

    def _decode_point_depth_conv(self, sequence):
        result = []
        first_depth = first_point = True
        pc = c = self.channels
        for i, token in enumerate(sequence):
            # compute output channels of this conv
            if i + 1 == len(sequence):
                assert token == "p", "Last conv must be point-wise conv."
                c = self.oup_main
            elif token == "p" and first_point:
                c = self.mid_channels
            if token == "d":
                # depth-wise conv
                assert pc == c, "Depth-wise conv must not change channels."
                result.append(nn.Conv2d(pc, c, self.ksize, self.stride if first_depth else 1, self.pad,
                                        groups=c, bias=False))
                result.append(nn.BatchNorm2d(c, affine=False))
                first_depth = False
            elif token == "p":
                # point-wise conv
                result.append(nn.Conv2d(pc, c, 1, 1, 0, bias=False))
                result.append(nn.BatchNorm2d(c, affine=False))
                result.append(nn.ReLU(inplace=True))
                first_point = False
            else:
                raise ValueError("Conv sequence must be d and p.")
            pc = c
        return result

    def _channel_shuffle(self, x):
        bs, num_channels, height, width = x.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(bs * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]

class ShuffleNetV2OneShot(nn.Module):
    block_keys = [
        'shufflenet_3x3',
        'shufflenet_5x5',
        'shufflenet_7x7',
        'xception_3x3',
    ]

    def __init__(self, input_size=224, first_conv_channels=16, last_conv_channels=1024, n_classes=1000):
        super().__init__()

        assert input_size % 32 == 0

        self.stage_blocks = [4, 4, 8, 4]
        self.stage_channels = [64, 160, 320, 640]
        self._input_size = input_size
        self._feature_map_size = input_size
        self._first_conv_channels = first_conv_channels
        self._last_conv_channels = last_conv_channels
        self._n_classes = n_classes

        # building first layer
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, first_conv_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(first_conv_channels, affine=False),
            nn.ReLU(inplace=True),
        )
        self._feature_map_size //= 2

        p_channels = first_conv_channels
        features = []
        for num_blocks, channels in zip(self.stage_blocks, self.stage_channels):
            features.extend(self._make_blocks(num_blocks, p_channels, channels))
            p_channels = channels
        self.features = nn.Sequential(*features)

        self.conv_last = nn.Sequential(
            nn.Conv2d(p_channels, last_conv_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(last_conv_channels, affine=False),
            nn.ReLU(inplace=True),
        )
        self.globalpool = nn.AvgPool2d(self._feature_map_size)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(last_conv_channels, n_classes, bias=False),
        )

        self._initialize_weights()

    def _make_blocks(self, blocks, in_channels, channels):
        #from nni.nas.pytorch import mutables
        result = []
        for i in range(blocks):
            stride = 2 if i == 0 else 1
            inp = in_channels if i == 0 else channels
            oup = channels

            base_mid_channels = channels // 2
            mid_channels = int(base_mid_channels)  # prepare for scale
            
            choice_block = ShuffleNetBlock(inp, oup, mid_channels=mid_channels, ksize=3, stride=stride) # to be mutated
            
            result.append(choice_block)

            if stride == 2:
                self._feature_map_size //= 2
        return result

    def forward(self, x):
        bs = x.size(0)
        x = self.first_conv(x)
        x = self.features(x)
        x = self.conv_last(x)
        x = self.globalpool(x)

        x = self.dropout(x)
        #x = x.contiguous().view(BATCH_SIZE, -1)
        x = x.contiguous().view(bs, -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

#====================Training approach

import sdk
from sdk.mutators.builtin_mutators import ModuleMutator
import datasets

class ModelTrain(sdk.Trainer):
    def __init__(self, device='cuda'):
        super(ModelTrain, self).__init__()
        self.device = torch.device(device)
        self.data_provider = datasets.ImagenetDataProvider(save_path="/data/v-yugzh/imagenet",
                                                    train_batch_size=32,
                                                    test_batch_size=32,
                                                    valid_size=None,
                                                    n_worker=4,
                                                    resize_scale=0.08,
                                                    distort_color='normal')

    def train_dataloader(self):
        return self.data_provider.train

    def val_dataloader(self):
        return self.data_provider.valid

# from spos import ShuffleNetV2OneShot, ModelTrain
# from sdk.mutators.builtin_mutators import ModuleMutator

# base_model = ShuffleNetV2OneShot()
# exp = sdk.create_experiment('mnist_search', base_model)
# exp.specify_training(ModelTrain)

# mutators = []
# for i in range(20):
#     mutator = ModuleMutator('features.'+str(i), [{'ksize': 3}, 
#                             {'ksize': 5}, {'ksize': 7}, 
#                             {'ksize': 3, 
#                             'sequence': 'dpdpdp'}])
#     mutators.append(mutator)
# exp.specify_mutators(mutators)
# exp.specify_strategy('naive.strategy.main', 'naive.strategy.RandomSampler')
# run_config = {
#     'authorName': 'nas',
#     'experimentName': 'nas',
#     'trialConcurrency': 1,
#     'maxExecDuration': '24h',
#     'maxTrialNum': 999,
#     'trainingServicePlatform': 'local',
#     'searchSpacePath': 'empty.json',
#     'useAnnotation': False
# } # nni experiment config
# pre_run_config = {
#     'name' : f'spos',
#     'x_shape' : [32, 3 , 244, 244],
#     'x_dtype' : 'torch.float32',
#     'y_shape' : [32],
#     "y_dtype" : "torch.int64",
#     "mask" : False
# }
# exp.run(run_config, pre_run_config=pre_run_config)