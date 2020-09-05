import torch
import torch.nn as nn
import torch.nn.functional as F

import sdk.custom_ops_torch as CUSTOM
from nni.nas.pytorch import mutables





class Graph(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers__0__conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=32, padding_mode='zeros')
        self.expand_dedup_physical_g0_layers__0__conv1 = CUSTOM.Expand(num_copies=192)
        self.expand_dedup_physical_g0__aten__relu_42 = CUSTOM.Expand(num_copies=192)
        self.layers__0__parallel_convs_BatchSizeView_pre = CUSTOM.BatchSizeView(batch_size=8)
        self.layers__0__parallel_convs = nn.Conv2d(in_channels=6144, out_channels=6144, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=192, padding_mode='zeros')
        self.layers__0__parallel_convs_BatchSizeView = CUSTOM.BatchSizeView(batch_size=1536)
        self.layers__0__conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, padding_mode='zeros')
        self.layers__1__conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=64, padding_mode='zeros')
        self.layers__1__parallel_convs_BatchSizeView_pre = CUSTOM.BatchSizeView(batch_size=8)
        self.layers__1__parallel_convs = nn.Conv2d(in_channels=12288, out_channels=12288, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=192, padding_mode='zeros')
        self.layers__1__parallel_convs_BatchSizeView = CUSTOM.BatchSizeView(batch_size=1536)
        self.layers__1__conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, padding_mode='zeros')
        self.layers__2__conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=128, padding_mode='zeros')
        self.layers__2__parallel_convs_BatchSizeView_pre = CUSTOM.BatchSizeView(batch_size=8)
        self.layers__2__parallel_convs = nn.Conv2d(in_channels=24576, out_channels=24576, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=192, padding_mode='zeros')
        self.layers__2__parallel_convs_BatchSizeView = CUSTOM.BatchSizeView(batch_size=1536)
        self.layers__2__conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, padding_mode='zeros')
        self.layers__3__conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=128, padding_mode='zeros')
        self.layers__3__parallel_convs_BatchSizeView_pre = CUSTOM.BatchSizeView(batch_size=8)
        self.layers__3__parallel_convs = nn.Conv2d(in_channels=24576, out_channels=24576, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=192, padding_mode='zeros')
        self.layers__3__parallel_convs_BatchSizeView = CUSTOM.BatchSizeView(batch_size=1536)
        self.layers__3__conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, padding_mode='zeros')
        self.layers__4__conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=256, padding_mode='zeros')
        self.layers__4__parallel_convs_BatchSizeView_pre = CUSTOM.BatchSizeView(batch_size=8)
        self.layers__4__parallel_convs = nn.Conv2d(in_channels=49152, out_channels=49152, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=192, padding_mode='zeros')
        self.layers__4__parallel_convs_BatchSizeView = CUSTOM.BatchSizeView(batch_size=1536)
        self.layers__4__conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, padding_mode='zeros')
        self.layers__5__conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=256, padding_mode='zeros')
        self.layers__5__parallel_convs_BatchSizeView_pre = CUSTOM.BatchSizeView(batch_size=8)
        self.layers__5__parallel_convs = nn.Conv2d(in_channels=49152, out_channels=49152, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=192, padding_mode='zeros')
        self.layers__5__parallel_convs_BatchSizeView = CUSTOM.BatchSizeView(batch_size=1536)
        self.layers__5__conv2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, padding_mode='zeros')
        self.layers__6__conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=512, padding_mode='zeros')
        self.layers__6__parallel_convs_BatchSizeView_pre = CUSTOM.BatchSizeView(batch_size=8)
        self.layers__6__parallel_convs = nn.Conv2d(in_channels=98304, out_channels=98304, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=192, padding_mode='zeros')
        self.layers__6__parallel_convs_BatchSizeView = CUSTOM.BatchSizeView(batch_size=1536)
        self.layers__6__conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, padding_mode='zeros')
        self.layers__7__conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=512, padding_mode='zeros')
        self.layers__7__parallel_convs_BatchSizeView_pre = CUSTOM.BatchSizeView(batch_size=8)
        self.layers__7__parallel_convs = nn.Conv2d(in_channels=98304, out_channels=98304, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=192, padding_mode='zeros')
        self.layers__7__parallel_convs_BatchSizeView = CUSTOM.BatchSizeView(batch_size=1536)
        self.layers__7__conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, padding_mode='zeros')
        self.layers__8__conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=512, padding_mode='zeros')
        self.layers__8__parallel_convs_BatchSizeView_pre = CUSTOM.BatchSizeView(batch_size=8)
        self.layers__8__parallel_convs = nn.Conv2d(in_channels=98304, out_channels=98304, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=192, padding_mode='zeros')
        self.layers__8__parallel_convs_BatchSizeView = CUSTOM.BatchSizeView(batch_size=1536)
        self.layers__8__conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, padding_mode='zeros')
        self.layers__9__conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=512, padding_mode='zeros')
        self.layers__9__parallel_convs_BatchSizeView_pre = CUSTOM.BatchSizeView(batch_size=8)
        self.layers__9__parallel_convs = nn.Conv2d(in_channels=98304, out_channels=98304, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=192, padding_mode='zeros')
        self.layers__9__parallel_convs_BatchSizeView = CUSTOM.BatchSizeView(batch_size=1536)
        self.layers__9__conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, padding_mode='zeros')
        self.layers__10__conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=512, padding_mode='zeros')
        self.layers__10__parallel_convs_BatchSizeView_pre = CUSTOM.BatchSizeView(batch_size=8)
        self.layers__10__parallel_convs = nn.Conv2d(in_channels=98304, out_channels=98304, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=192, padding_mode='zeros')
        self.layers__10__parallel_convs_BatchSizeView = CUSTOM.BatchSizeView(batch_size=1536)
        self.layers__10__conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, padding_mode='zeros')
        self.layers__11__conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=512, padding_mode='zeros')
        self.layers__11__parallel_convs_BatchSizeView_pre = CUSTOM.BatchSizeView(batch_size=8)
        self.layers__11__parallel_convs = nn.Conv2d(in_channels=98304, out_channels=98304, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=192, padding_mode='zeros')
        self.layers__11__parallel_convs_BatchSizeView = CUSTOM.BatchSizeView(batch_size=1536)
        self.layers__11__conv2 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, padding_mode='zeros')
        self.layers__12__conv1 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1024, padding_mode='zeros')
        self.layers__12__parallel_convs_BatchSizeView_pre = CUSTOM.BatchSizeView(batch_size=8)
        self.layers__12__parallel_convs = nn.Conv2d(in_channels=196608, out_channels=196608, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=192, padding_mode='zeros')
        self.layers__12__parallel_convs_BatchSizeView = CUSTOM.BatchSizeView(batch_size=1536)
        self.layers__12__conv2 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, padding_mode='zeros')
        self.linear = nn.Linear(in_features=1024, out_features=10, bias=True)
        self.expand_dedup_physical_g0_input_1 = CUSTOM.Expand(num_copies=192)
        self.expand_dedup_physical_g0_input_2 = CUSTOM.Expand(num_copies=192)

    def forward(self, input_1, input_2):
        input_1 = input_1.cuda(non_blocking=True)
        input_2 = input_2.cuda(non_blocking=True)
        conv1 = self.conv1(input_1)
        bn1 = self.bn1(conv1)
        del conv1
        _aten__relu_42 = F.relu(bn1, )
        del bn1
        layers__0__conv1 = self.layers__0__conv1(_aten__relu_42)
        layers_0_aten__add_47_input_dedup_physical_g0_layers__0__conv1 = layers__0__conv1
        del layers__0__conv1
        expand_dedup_physical_g0_layers__0__conv1 = self.expand_dedup_physical_g0_layers__0__conv1(layers_0_aten__add_47_input_dedup_physical_g0_layers__0__conv1)
        del layers_0_aten__add_47_input_dedup_physical_g0_layers__0__conv1
        layers__0__parallel_convs_input_dedup_physical_g0__aten__relu_42 = _aten__relu_42
        del _aten__relu_42
        expand_dedup_physical_g0__aten__relu_42 = self.expand_dedup_physical_g0__aten__relu_42(layers__0__parallel_convs_input_dedup_physical_g0__aten__relu_42)
        del layers__0__parallel_convs_input_dedup_physical_g0__aten__relu_42
        layers__0__parallel_convs_BatchSizeView_pre = self.layers__0__parallel_convs_BatchSizeView_pre(expand_dedup_physical_g0__aten__relu_42)
        del expand_dedup_physical_g0__aten__relu_42
        layers__0__parallel_convs = self.layers__0__parallel_convs(layers__0__parallel_convs_BatchSizeView_pre)
        del layers__0__parallel_convs_BatchSizeView_pre
        layers__0__parallel_convs_BatchSizeView = self.layers__0__parallel_convs_BatchSizeView(layers__0__parallel_convs)
        del layers__0__parallel_convs
        layers__0__parallel_convs_output = layers__0__parallel_convs_BatchSizeView
        del layers__0__parallel_convs_BatchSizeView
        layers_0_aten__add_47_input_batch_physical_g0_layers__0__parallel_convs = layers__0__parallel_convs_output
        del layers__0__parallel_convs_output
        layers_0_aten__add_47 = expand_dedup_physical_g0_layers__0__conv1 + layers_0_aten__add_47_input_batch_physical_g0_layers__0__parallel_convs
        del expand_dedup_physical_g0_layers__0__conv1
        del layers_0_aten__add_47_input_batch_physical_g0_layers__0__parallel_convs
        layers_0_aten__add_47_output = layers_0_aten__add_47
        del layers_0_aten__add_47
        layers_0_aten__relu_48_input_batch_physical_g0_layers_0_aten__add_47 = layers_0_aten__add_47_output
        del layers_0_aten__add_47_output
        layers_0_aten__relu_48 = F.relu(layers_0_aten__relu_48_input_batch_physical_g0_layers_0_aten__add_47, )
        del layers_0_aten__relu_48_input_batch_physical_g0_layers_0_aten__add_47
        layers_0_aten__relu_48_output = layers_0_aten__relu_48
        del layers_0_aten__relu_48
        layers__0__conv2_input_batch_physical_g0_layers_0_aten__relu_48 = layers_0_aten__relu_48_output
        del layers_0_aten__relu_48_output
        layers__0__conv2 = self.layers__0__conv2(layers__0__conv2_input_batch_physical_g0_layers_0_aten__relu_48)
        del layers__0__conv2_input_batch_physical_g0_layers_0_aten__relu_48
        layers__0__conv2_output = layers__0__conv2
        del layers__0__conv2
        layers_0_aten__relu_49_input_batch_physical_g0_layers__0__conv2 = layers__0__conv2_output
        del layers__0__conv2_output
        layers_0_aten__relu_49 = F.relu(layers_0_aten__relu_49_input_batch_physical_g0_layers__0__conv2, )
        del layers_0_aten__relu_49_input_batch_physical_g0_layers__0__conv2
        layers_0_aten__relu_49_output = layers_0_aten__relu_49
        del layers_0_aten__relu_49
        layers__1__conv1_input_batch_physical_g0_layers_0_aten__relu_49 = layers_0_aten__relu_49_output
        layers__1__conv1 = self.layers__1__conv1(layers__1__conv1_input_batch_physical_g0_layers_0_aten__relu_49)
        del layers__1__conv1_input_batch_physical_g0_layers_0_aten__relu_49
        layers__1__conv1_output = layers__1__conv1
        del layers__1__conv1
        layers_1_aten__add_50_input_batch_physical_g0_layers__1__conv1 = layers__1__conv1_output
        del layers__1__conv1_output
        layers__1__parallel_convs_input_batch_physical_g0_layers_0_aten__relu_49 = layers_0_aten__relu_49_output
        del layers_0_aten__relu_49_output
        layers__1__parallel_convs_BatchSizeView_pre = self.layers__1__parallel_convs_BatchSizeView_pre(layers__1__parallel_convs_input_batch_physical_g0_layers_0_aten__relu_49)
        del layers__1__parallel_convs_input_batch_physical_g0_layers_0_aten__relu_49
        layers__1__parallel_convs = self.layers__1__parallel_convs(layers__1__parallel_convs_BatchSizeView_pre)
        del layers__1__parallel_convs_BatchSizeView_pre
        layers__1__parallel_convs_BatchSizeView = self.layers__1__parallel_convs_BatchSizeView(layers__1__parallel_convs)
        del layers__1__parallel_convs
        layers__1__parallel_convs_output = layers__1__parallel_convs_BatchSizeView
        del layers__1__parallel_convs_BatchSizeView
        layers_1_aten__add_50_input_batch_physical_g0_layers__1__parallel_convs = layers__1__parallel_convs_output
        del layers__1__parallel_convs_output
        layers_1_aten__add_50 = layers_1_aten__add_50_input_batch_physical_g0_layers__1__conv1 + layers_1_aten__add_50_input_batch_physical_g0_layers__1__parallel_convs
        del layers_1_aten__add_50_input_batch_physical_g0_layers__1__conv1
        del layers_1_aten__add_50_input_batch_physical_g0_layers__1__parallel_convs
        layers_1_aten__add_50_output = layers_1_aten__add_50
        del layers_1_aten__add_50
        layers_1_aten__relu_51_input_batch_physical_g0_layers_1_aten__add_50 = layers_1_aten__add_50_output
        del layers_1_aten__add_50_output
        layers_1_aten__relu_51 = F.relu(layers_1_aten__relu_51_input_batch_physical_g0_layers_1_aten__add_50, )
        del layers_1_aten__relu_51_input_batch_physical_g0_layers_1_aten__add_50
        layers_1_aten__relu_51_output = layers_1_aten__relu_51
        del layers_1_aten__relu_51
        layers__1__conv2_input_batch_physical_g0_layers_1_aten__relu_51 = layers_1_aten__relu_51_output
        del layers_1_aten__relu_51_output
        layers__1__conv2 = self.layers__1__conv2(layers__1__conv2_input_batch_physical_g0_layers_1_aten__relu_51)
        del layers__1__conv2_input_batch_physical_g0_layers_1_aten__relu_51
        layers__1__conv2_output = layers__1__conv2
        del layers__1__conv2
        layers_1_aten__relu_52_input_batch_physical_g0_layers__1__conv2 = layers__1__conv2_output
        del layers__1__conv2_output
        layers_1_aten__relu_52 = F.relu(layers_1_aten__relu_52_input_batch_physical_g0_layers__1__conv2, )
        del layers_1_aten__relu_52_input_batch_physical_g0_layers__1__conv2
        layers_1_aten__relu_52_output = layers_1_aten__relu_52
        del layers_1_aten__relu_52
        layers__2__conv1_input_batch_physical_g0_layers_1_aten__relu_52 = layers_1_aten__relu_52_output
        layers__2__conv1 = self.layers__2__conv1(layers__2__conv1_input_batch_physical_g0_layers_1_aten__relu_52)
        del layers__2__conv1_input_batch_physical_g0_layers_1_aten__relu_52
        layers__2__conv1_output = layers__2__conv1
        del layers__2__conv1
        layers_2_aten__add_53_input_batch_physical_g0_layers__2__conv1 = layers__2__conv1_output
        del layers__2__conv1_output
        layers__2__parallel_convs_input_batch_physical_g0_layers_1_aten__relu_52 = layers_1_aten__relu_52_output
        del layers_1_aten__relu_52_output
        layers__2__parallel_convs_BatchSizeView_pre = self.layers__2__parallel_convs_BatchSizeView_pre(layers__2__parallel_convs_input_batch_physical_g0_layers_1_aten__relu_52)
        del layers__2__parallel_convs_input_batch_physical_g0_layers_1_aten__relu_52
        layers__2__parallel_convs = self.layers__2__parallel_convs(layers__2__parallel_convs_BatchSizeView_pre)
        del layers__2__parallel_convs_BatchSizeView_pre
        layers__2__parallel_convs_BatchSizeView = self.layers__2__parallel_convs_BatchSizeView(layers__2__parallel_convs)
        del layers__2__parallel_convs
        layers__2__parallel_convs_output = layers__2__parallel_convs_BatchSizeView
        del layers__2__parallel_convs_BatchSizeView
        layers_2_aten__add_53_input_batch_physical_g0_layers__2__parallel_convs = layers__2__parallel_convs_output
        del layers__2__parallel_convs_output
        layers_2_aten__add_53 = layers_2_aten__add_53_input_batch_physical_g0_layers__2__conv1 + layers_2_aten__add_53_input_batch_physical_g0_layers__2__parallel_convs
        del layers_2_aten__add_53_input_batch_physical_g0_layers__2__conv1
        del layers_2_aten__add_53_input_batch_physical_g0_layers__2__parallel_convs
        layers_2_aten__add_53_output = layers_2_aten__add_53
        del layers_2_aten__add_53
        layers_2_aten__relu_54_input_batch_physical_g0_layers_2_aten__add_53 = layers_2_aten__add_53_output
        del layers_2_aten__add_53_output
        layers_2_aten__relu_54 = F.relu(layers_2_aten__relu_54_input_batch_physical_g0_layers_2_aten__add_53, )
        del layers_2_aten__relu_54_input_batch_physical_g0_layers_2_aten__add_53
        layers_2_aten__relu_54_output = layers_2_aten__relu_54
        del layers_2_aten__relu_54
        layers__2__conv2_input_batch_physical_g0_layers_2_aten__relu_54 = layers_2_aten__relu_54_output
        del layers_2_aten__relu_54_output
        layers__2__conv2 = self.layers__2__conv2(layers__2__conv2_input_batch_physical_g0_layers_2_aten__relu_54)
        del layers__2__conv2_input_batch_physical_g0_layers_2_aten__relu_54
        layers__2__conv2_output = layers__2__conv2
        del layers__2__conv2
        layers_2_aten__relu_55_input_batch_physical_g0_layers__2__conv2 = layers__2__conv2_output
        del layers__2__conv2_output
        layers_2_aten__relu_55 = F.relu(layers_2_aten__relu_55_input_batch_physical_g0_layers__2__conv2, )
        del layers_2_aten__relu_55_input_batch_physical_g0_layers__2__conv2
        layers_2_aten__relu_55_output = layers_2_aten__relu_55
        del layers_2_aten__relu_55
        layers__3__conv1_input_batch_physical_g0_layers_2_aten__relu_55 = layers_2_aten__relu_55_output
        layers__3__conv1 = self.layers__3__conv1(layers__3__conv1_input_batch_physical_g0_layers_2_aten__relu_55)
        del layers__3__conv1_input_batch_physical_g0_layers_2_aten__relu_55
        layers__3__conv1_output = layers__3__conv1
        del layers__3__conv1
        layers_3_aten__add_56_input_batch_physical_g0_layers__3__conv1 = layers__3__conv1_output
        del layers__3__conv1_output
        layers__3__parallel_convs_input_batch_physical_g0_layers_2_aten__relu_55 = layers_2_aten__relu_55_output
        del layers_2_aten__relu_55_output
        layers__3__parallel_convs_BatchSizeView_pre = self.layers__3__parallel_convs_BatchSizeView_pre(layers__3__parallel_convs_input_batch_physical_g0_layers_2_aten__relu_55)
        del layers__3__parallel_convs_input_batch_physical_g0_layers_2_aten__relu_55
        layers__3__parallel_convs = self.layers__3__parallel_convs(layers__3__parallel_convs_BatchSizeView_pre)
        del layers__3__parallel_convs_BatchSizeView_pre
        layers__3__parallel_convs_BatchSizeView = self.layers__3__parallel_convs_BatchSizeView(layers__3__parallel_convs)
        del layers__3__parallel_convs
        layers__3__parallel_convs_output = layers__3__parallel_convs_BatchSizeView
        del layers__3__parallel_convs_BatchSizeView
        layers_3_aten__add_56_input_batch_physical_g0_layers__3__parallel_convs = layers__3__parallel_convs_output
        del layers__3__parallel_convs_output
        layers_3_aten__add_56 = layers_3_aten__add_56_input_batch_physical_g0_layers__3__conv1 + layers_3_aten__add_56_input_batch_physical_g0_layers__3__parallel_convs
        del layers_3_aten__add_56_input_batch_physical_g0_layers__3__conv1
        del layers_3_aten__add_56_input_batch_physical_g0_layers__3__parallel_convs
        layers_3_aten__add_56_output = layers_3_aten__add_56
        del layers_3_aten__add_56
        layers_3_aten__relu_57_input_batch_physical_g0_layers_3_aten__add_56 = layers_3_aten__add_56_output
        del layers_3_aten__add_56_output
        layers_3_aten__relu_57 = F.relu(layers_3_aten__relu_57_input_batch_physical_g0_layers_3_aten__add_56, )
        del layers_3_aten__relu_57_input_batch_physical_g0_layers_3_aten__add_56
        layers_3_aten__relu_57_output = layers_3_aten__relu_57
        del layers_3_aten__relu_57
        layers__3__conv2_input_batch_physical_g0_layers_3_aten__relu_57 = layers_3_aten__relu_57_output
        del layers_3_aten__relu_57_output
        layers__3__conv2 = self.layers__3__conv2(layers__3__conv2_input_batch_physical_g0_layers_3_aten__relu_57)
        del layers__3__conv2_input_batch_physical_g0_layers_3_aten__relu_57
        layers__3__conv2_output = layers__3__conv2
        del layers__3__conv2
        layers_3_aten__relu_58_input_batch_physical_g0_layers__3__conv2 = layers__3__conv2_output
        del layers__3__conv2_output
        layers_3_aten__relu_58 = F.relu(layers_3_aten__relu_58_input_batch_physical_g0_layers__3__conv2, )
        del layers_3_aten__relu_58_input_batch_physical_g0_layers__3__conv2
        layers_3_aten__relu_58_output = layers_3_aten__relu_58
        del layers_3_aten__relu_58
        layers__4__conv1_input_batch_physical_g0_layers_3_aten__relu_58 = layers_3_aten__relu_58_output
        layers__4__conv1 = self.layers__4__conv1(layers__4__conv1_input_batch_physical_g0_layers_3_aten__relu_58)
        del layers__4__conv1_input_batch_physical_g0_layers_3_aten__relu_58
        layers__4__conv1_output = layers__4__conv1
        del layers__4__conv1
        layers_4_aten__add_59_input_batch_physical_g0_layers__4__conv1 = layers__4__conv1_output
        del layers__4__conv1_output
        layers__4__parallel_convs_input_batch_physical_g0_layers_3_aten__relu_58 = layers_3_aten__relu_58_output
        del layers_3_aten__relu_58_output
        layers__4__parallel_convs_BatchSizeView_pre = self.layers__4__parallel_convs_BatchSizeView_pre(layers__4__parallel_convs_input_batch_physical_g0_layers_3_aten__relu_58)
        del layers__4__parallel_convs_input_batch_physical_g0_layers_3_aten__relu_58
        layers__4__parallel_convs = self.layers__4__parallel_convs(layers__4__parallel_convs_BatchSizeView_pre)
        del layers__4__parallel_convs_BatchSizeView_pre
        layers__4__parallel_convs_BatchSizeView = self.layers__4__parallel_convs_BatchSizeView(layers__4__parallel_convs)
        del layers__4__parallel_convs
        layers__4__parallel_convs_output = layers__4__parallel_convs_BatchSizeView
        del layers__4__parallel_convs_BatchSizeView
        layers_4_aten__add_59_input_batch_physical_g0_layers__4__parallel_convs = layers__4__parallel_convs_output
        del layers__4__parallel_convs_output
        layers_4_aten__add_59 = layers_4_aten__add_59_input_batch_physical_g0_layers__4__conv1 + layers_4_aten__add_59_input_batch_physical_g0_layers__4__parallel_convs
        del layers_4_aten__add_59_input_batch_physical_g0_layers__4__conv1
        del layers_4_aten__add_59_input_batch_physical_g0_layers__4__parallel_convs
        layers_4_aten__add_59_output = layers_4_aten__add_59
        del layers_4_aten__add_59
        layers_4_aten__relu_60_input_batch_physical_g0_layers_4_aten__add_59 = layers_4_aten__add_59_output
        del layers_4_aten__add_59_output
        layers_4_aten__relu_60 = F.relu(layers_4_aten__relu_60_input_batch_physical_g0_layers_4_aten__add_59, )
        del layers_4_aten__relu_60_input_batch_physical_g0_layers_4_aten__add_59
        layers_4_aten__relu_60_output = layers_4_aten__relu_60
        del layers_4_aten__relu_60
        layers__4__conv2_input_batch_physical_g0_layers_4_aten__relu_60 = layers_4_aten__relu_60_output
        del layers_4_aten__relu_60_output
        layers__4__conv2 = self.layers__4__conv2(layers__4__conv2_input_batch_physical_g0_layers_4_aten__relu_60)
        del layers__4__conv2_input_batch_physical_g0_layers_4_aten__relu_60
        layers__4__conv2_output = layers__4__conv2
        del layers__4__conv2
        layers_4_aten__relu_61_input_batch_physical_g0_layers__4__conv2 = layers__4__conv2_output
        del layers__4__conv2_output
        layers_4_aten__relu_61 = F.relu(layers_4_aten__relu_61_input_batch_physical_g0_layers__4__conv2, )
        del layers_4_aten__relu_61_input_batch_physical_g0_layers__4__conv2
        layers_4_aten__relu_61_output = layers_4_aten__relu_61
        del layers_4_aten__relu_61
        layers__5__conv1_input_batch_physical_g0_layers_4_aten__relu_61 = layers_4_aten__relu_61_output
        layers__5__conv1 = self.layers__5__conv1(layers__5__conv1_input_batch_physical_g0_layers_4_aten__relu_61)
        del layers__5__conv1_input_batch_physical_g0_layers_4_aten__relu_61
        layers__5__conv1_output = layers__5__conv1
        del layers__5__conv1
        layers_5_aten__add_62_input_batch_physical_g0_layers__5__conv1 = layers__5__conv1_output
        del layers__5__conv1_output
        layers__5__parallel_convs_input_batch_physical_g0_layers_4_aten__relu_61 = layers_4_aten__relu_61_output
        del layers_4_aten__relu_61_output
        layers__5__parallel_convs_BatchSizeView_pre = self.layers__5__parallel_convs_BatchSizeView_pre(layers__5__parallel_convs_input_batch_physical_g0_layers_4_aten__relu_61)
        del layers__5__parallel_convs_input_batch_physical_g0_layers_4_aten__relu_61
        layers__5__parallel_convs = self.layers__5__parallel_convs(layers__5__parallel_convs_BatchSizeView_pre)
        del layers__5__parallel_convs_BatchSizeView_pre
        layers__5__parallel_convs_BatchSizeView = self.layers__5__parallel_convs_BatchSizeView(layers__5__parallel_convs)
        del layers__5__parallel_convs
        layers__5__parallel_convs_output = layers__5__parallel_convs_BatchSizeView
        del layers__5__parallel_convs_BatchSizeView
        layers_5_aten__add_62_input_batch_physical_g0_layers__5__parallel_convs = layers__5__parallel_convs_output
        del layers__5__parallel_convs_output
        layers_5_aten__add_62 = layers_5_aten__add_62_input_batch_physical_g0_layers__5__conv1 + layers_5_aten__add_62_input_batch_physical_g0_layers__5__parallel_convs
        del layers_5_aten__add_62_input_batch_physical_g0_layers__5__conv1
        del layers_5_aten__add_62_input_batch_physical_g0_layers__5__parallel_convs
        layers_5_aten__add_62_output = layers_5_aten__add_62
        del layers_5_aten__add_62
        layers_5_aten__relu_63_input_batch_physical_g0_layers_5_aten__add_62 = layers_5_aten__add_62_output
        del layers_5_aten__add_62_output
        layers_5_aten__relu_63 = F.relu(layers_5_aten__relu_63_input_batch_physical_g0_layers_5_aten__add_62, )
        del layers_5_aten__relu_63_input_batch_physical_g0_layers_5_aten__add_62
        layers_5_aten__relu_63_output = layers_5_aten__relu_63
        del layers_5_aten__relu_63
        layers__5__conv2_input_batch_physical_g0_layers_5_aten__relu_63 = layers_5_aten__relu_63_output
        del layers_5_aten__relu_63_output
        layers__5__conv2 = self.layers__5__conv2(layers__5__conv2_input_batch_physical_g0_layers_5_aten__relu_63)
        del layers__5__conv2_input_batch_physical_g0_layers_5_aten__relu_63
        layers__5__conv2_output = layers__5__conv2
        del layers__5__conv2
        layers_5_aten__relu_64_input_batch_physical_g0_layers__5__conv2 = layers__5__conv2_output
        del layers__5__conv2_output
        layers_5_aten__relu_64 = F.relu(layers_5_aten__relu_64_input_batch_physical_g0_layers__5__conv2, )
        del layers_5_aten__relu_64_input_batch_physical_g0_layers__5__conv2
        layers_5_aten__relu_64_output = layers_5_aten__relu_64
        del layers_5_aten__relu_64
        layers__6__conv1_input_batch_physical_g0_layers_5_aten__relu_64 = layers_5_aten__relu_64_output
        layers__6__conv1 = self.layers__6__conv1(layers__6__conv1_input_batch_physical_g0_layers_5_aten__relu_64)
        del layers__6__conv1_input_batch_physical_g0_layers_5_aten__relu_64
        layers__6__conv1_output = layers__6__conv1
        del layers__6__conv1
        layers_6_aten__add_65_input_batch_physical_g0_layers__6__conv1 = layers__6__conv1_output
        del layers__6__conv1_output
        layers__6__parallel_convs_input_batch_physical_g0_layers_5_aten__relu_64 = layers_5_aten__relu_64_output
        del layers_5_aten__relu_64_output
        layers__6__parallel_convs_BatchSizeView_pre = self.layers__6__parallel_convs_BatchSizeView_pre(layers__6__parallel_convs_input_batch_physical_g0_layers_5_aten__relu_64)
        del layers__6__parallel_convs_input_batch_physical_g0_layers_5_aten__relu_64
        layers__6__parallel_convs = self.layers__6__parallel_convs(layers__6__parallel_convs_BatchSizeView_pre)
        del layers__6__parallel_convs_BatchSizeView_pre
        layers__6__parallel_convs_BatchSizeView = self.layers__6__parallel_convs_BatchSizeView(layers__6__parallel_convs)
        del layers__6__parallel_convs
        layers__6__parallel_convs_output = layers__6__parallel_convs_BatchSizeView
        del layers__6__parallel_convs_BatchSizeView
        layers_6_aten__add_65_input_batch_physical_g0_layers__6__parallel_convs = layers__6__parallel_convs_output
        del layers__6__parallel_convs_output
        layers_6_aten__add_65 = layers_6_aten__add_65_input_batch_physical_g0_layers__6__conv1 + layers_6_aten__add_65_input_batch_physical_g0_layers__6__parallel_convs
        del layers_6_aten__add_65_input_batch_physical_g0_layers__6__conv1
        del layers_6_aten__add_65_input_batch_physical_g0_layers__6__parallel_convs
        layers_6_aten__add_65_output = layers_6_aten__add_65
        del layers_6_aten__add_65
        layers_6_aten__relu_66_input_batch_physical_g0_layers_6_aten__add_65 = layers_6_aten__add_65_output
        del layers_6_aten__add_65_output
        layers_6_aten__relu_66 = F.relu(layers_6_aten__relu_66_input_batch_physical_g0_layers_6_aten__add_65, )
        del layers_6_aten__relu_66_input_batch_physical_g0_layers_6_aten__add_65
        layers_6_aten__relu_66_output = layers_6_aten__relu_66
        del layers_6_aten__relu_66
        layers__6__conv2_input_batch_physical_g0_layers_6_aten__relu_66 = layers_6_aten__relu_66_output
        del layers_6_aten__relu_66_output
        layers__6__conv2 = self.layers__6__conv2(layers__6__conv2_input_batch_physical_g0_layers_6_aten__relu_66)
        del layers__6__conv2_input_batch_physical_g0_layers_6_aten__relu_66
        layers__6__conv2_output = layers__6__conv2
        del layers__6__conv2
        layers_6_aten__relu_67_input_batch_physical_g0_layers__6__conv2 = layers__6__conv2_output
        del layers__6__conv2_output
        layers_6_aten__relu_67 = F.relu(layers_6_aten__relu_67_input_batch_physical_g0_layers__6__conv2, )
        del layers_6_aten__relu_67_input_batch_physical_g0_layers__6__conv2
        layers_6_aten__relu_67_output = layers_6_aten__relu_67
        del layers_6_aten__relu_67
        layers__7__conv1_input_batch_physical_g0_layers_6_aten__relu_67 = layers_6_aten__relu_67_output
        layers__7__conv1 = self.layers__7__conv1(layers__7__conv1_input_batch_physical_g0_layers_6_aten__relu_67)
        del layers__7__conv1_input_batch_physical_g0_layers_6_aten__relu_67
        layers__7__conv1_output = layers__7__conv1
        del layers__7__conv1
        layers_7_aten__add_68_input_batch_physical_g0_layers__7__conv1 = layers__7__conv1_output
        del layers__7__conv1_output
        layers__7__parallel_convs_input_batch_physical_g0_layers_6_aten__relu_67 = layers_6_aten__relu_67_output
        del layers_6_aten__relu_67_output
        layers__7__parallel_convs_BatchSizeView_pre = self.layers__7__parallel_convs_BatchSizeView_pre(layers__7__parallel_convs_input_batch_physical_g0_layers_6_aten__relu_67)
        del layers__7__parallel_convs_input_batch_physical_g0_layers_6_aten__relu_67
        layers__7__parallel_convs = self.layers__7__parallel_convs(layers__7__parallel_convs_BatchSizeView_pre)
        del layers__7__parallel_convs_BatchSizeView_pre
        layers__7__parallel_convs_BatchSizeView = self.layers__7__parallel_convs_BatchSizeView(layers__7__parallel_convs)
        del layers__7__parallel_convs
        layers__7__parallel_convs_output = layers__7__parallel_convs_BatchSizeView
        del layers__7__parallel_convs_BatchSizeView
        layers_7_aten__add_68_input_batch_physical_g0_layers__7__parallel_convs = layers__7__parallel_convs_output
        del layers__7__parallel_convs_output
        layers_7_aten__add_68 = layers_7_aten__add_68_input_batch_physical_g0_layers__7__conv1 + layers_7_aten__add_68_input_batch_physical_g0_layers__7__parallel_convs
        del layers_7_aten__add_68_input_batch_physical_g0_layers__7__conv1
        del layers_7_aten__add_68_input_batch_physical_g0_layers__7__parallel_convs
        layers_7_aten__add_68_output = layers_7_aten__add_68
        del layers_7_aten__add_68
        layers_7_aten__relu_69_input_batch_physical_g0_layers_7_aten__add_68 = layers_7_aten__add_68_output
        del layers_7_aten__add_68_output
        layers_7_aten__relu_69 = F.relu(layers_7_aten__relu_69_input_batch_physical_g0_layers_7_aten__add_68, )
        del layers_7_aten__relu_69_input_batch_physical_g0_layers_7_aten__add_68
        layers_7_aten__relu_69_output = layers_7_aten__relu_69
        del layers_7_aten__relu_69
        layers__7__conv2_input_batch_physical_g0_layers_7_aten__relu_69 = layers_7_aten__relu_69_output
        del layers_7_aten__relu_69_output
        layers__7__conv2 = self.layers__7__conv2(layers__7__conv2_input_batch_physical_g0_layers_7_aten__relu_69)
        del layers__7__conv2_input_batch_physical_g0_layers_7_aten__relu_69
        layers__7__conv2_output = layers__7__conv2
        del layers__7__conv2
        layers_7_aten__relu_70_input_batch_physical_g0_layers__7__conv2 = layers__7__conv2_output
        del layers__7__conv2_output
        layers_7_aten__relu_70 = F.relu(layers_7_aten__relu_70_input_batch_physical_g0_layers__7__conv2, )
        del layers_7_aten__relu_70_input_batch_physical_g0_layers__7__conv2
        layers_7_aten__relu_70_output = layers_7_aten__relu_70
        del layers_7_aten__relu_70
        layers__8__conv1_input_batch_physical_g0_layers_7_aten__relu_70 = layers_7_aten__relu_70_output
        layers__8__conv1 = self.layers__8__conv1(layers__8__conv1_input_batch_physical_g0_layers_7_aten__relu_70)
        del layers__8__conv1_input_batch_physical_g0_layers_7_aten__relu_70
        layers__8__conv1_output = layers__8__conv1
        del layers__8__conv1
        layers_8_aten__add_71_input_batch_physical_g0_layers__8__conv1 = layers__8__conv1_output
        del layers__8__conv1_output
        layers__8__parallel_convs_input_batch_physical_g0_layers_7_aten__relu_70 = layers_7_aten__relu_70_output
        del layers_7_aten__relu_70_output
        layers__8__parallel_convs_BatchSizeView_pre = self.layers__8__parallel_convs_BatchSizeView_pre(layers__8__parallel_convs_input_batch_physical_g0_layers_7_aten__relu_70)
        del layers__8__parallel_convs_input_batch_physical_g0_layers_7_aten__relu_70
        layers__8__parallel_convs = self.layers__8__parallel_convs(layers__8__parallel_convs_BatchSizeView_pre)
        del layers__8__parallel_convs_BatchSizeView_pre
        layers__8__parallel_convs_BatchSizeView = self.layers__8__parallel_convs_BatchSizeView(layers__8__parallel_convs)
        del layers__8__parallel_convs
        layers__8__parallel_convs_output = layers__8__parallel_convs_BatchSizeView
        del layers__8__parallel_convs_BatchSizeView
        layers_8_aten__add_71_input_batch_physical_g0_layers__8__parallel_convs = layers__8__parallel_convs_output
        del layers__8__parallel_convs_output
        layers_8_aten__add_71 = layers_8_aten__add_71_input_batch_physical_g0_layers__8__conv1 + layers_8_aten__add_71_input_batch_physical_g0_layers__8__parallel_convs
        del layers_8_aten__add_71_input_batch_physical_g0_layers__8__conv1
        del layers_8_aten__add_71_input_batch_physical_g0_layers__8__parallel_convs
        layers_8_aten__add_71_output = layers_8_aten__add_71
        del layers_8_aten__add_71
        layers_8_aten__relu_72_input_batch_physical_g0_layers_8_aten__add_71 = layers_8_aten__add_71_output
        del layers_8_aten__add_71_output
        layers_8_aten__relu_72 = F.relu(layers_8_aten__relu_72_input_batch_physical_g0_layers_8_aten__add_71, )
        del layers_8_aten__relu_72_input_batch_physical_g0_layers_8_aten__add_71
        layers_8_aten__relu_72_output = layers_8_aten__relu_72
        del layers_8_aten__relu_72
        layers__8__conv2_input_batch_physical_g0_layers_8_aten__relu_72 = layers_8_aten__relu_72_output
        del layers_8_aten__relu_72_output
        layers__8__conv2 = self.layers__8__conv2(layers__8__conv2_input_batch_physical_g0_layers_8_aten__relu_72)
        del layers__8__conv2_input_batch_physical_g0_layers_8_aten__relu_72
        layers__8__conv2_output = layers__8__conv2
        del layers__8__conv2
        layers_8_aten__relu_73_input_batch_physical_g0_layers__8__conv2 = layers__8__conv2_output
        del layers__8__conv2_output
        layers_8_aten__relu_73 = F.relu(layers_8_aten__relu_73_input_batch_physical_g0_layers__8__conv2, )
        del layers_8_aten__relu_73_input_batch_physical_g0_layers__8__conv2
        layers_8_aten__relu_73_output = layers_8_aten__relu_73
        del layers_8_aten__relu_73
        layers__9__conv1_input_batch_physical_g0_layers_8_aten__relu_73 = layers_8_aten__relu_73_output
        layers__9__conv1 = self.layers__9__conv1(layers__9__conv1_input_batch_physical_g0_layers_8_aten__relu_73)
        del layers__9__conv1_input_batch_physical_g0_layers_8_aten__relu_73
        layers__9__conv1_output = layers__9__conv1
        del layers__9__conv1
        layers_9_aten__add_74_input_batch_physical_g0_layers__9__conv1 = layers__9__conv1_output
        del layers__9__conv1_output
        layers__9__parallel_convs_input_batch_physical_g0_layers_8_aten__relu_73 = layers_8_aten__relu_73_output
        del layers_8_aten__relu_73_output
        layers__9__parallel_convs_BatchSizeView_pre = self.layers__9__parallel_convs_BatchSizeView_pre(layers__9__parallel_convs_input_batch_physical_g0_layers_8_aten__relu_73)
        del layers__9__parallel_convs_input_batch_physical_g0_layers_8_aten__relu_73
        layers__9__parallel_convs = self.layers__9__parallel_convs(layers__9__parallel_convs_BatchSizeView_pre)
        del layers__9__parallel_convs_BatchSizeView_pre
        layers__9__parallel_convs_BatchSizeView = self.layers__9__parallel_convs_BatchSizeView(layers__9__parallel_convs)
        del layers__9__parallel_convs
        layers__9__parallel_convs_output = layers__9__parallel_convs_BatchSizeView
        del layers__9__parallel_convs_BatchSizeView
        layers_9_aten__add_74_input_batch_physical_g0_layers__9__parallel_convs = layers__9__parallel_convs_output
        del layers__9__parallel_convs_output
        layers_9_aten__add_74 = layers_9_aten__add_74_input_batch_physical_g0_layers__9__conv1 + layers_9_aten__add_74_input_batch_physical_g0_layers__9__parallel_convs
        del layers_9_aten__add_74_input_batch_physical_g0_layers__9__conv1
        del layers_9_aten__add_74_input_batch_physical_g0_layers__9__parallel_convs
        layers_9_aten__add_74_output = layers_9_aten__add_74
        del layers_9_aten__add_74
        layers_9_aten__relu_75_input_batch_physical_g0_layers_9_aten__add_74 = layers_9_aten__add_74_output
        del layers_9_aten__add_74_output
        layers_9_aten__relu_75 = F.relu(layers_9_aten__relu_75_input_batch_physical_g0_layers_9_aten__add_74, )
        del layers_9_aten__relu_75_input_batch_physical_g0_layers_9_aten__add_74
        layers_9_aten__relu_75_output = layers_9_aten__relu_75
        del layers_9_aten__relu_75
        layers__9__conv2_input_batch_physical_g0_layers_9_aten__relu_75 = layers_9_aten__relu_75_output
        del layers_9_aten__relu_75_output
        layers__9__conv2 = self.layers__9__conv2(layers__9__conv2_input_batch_physical_g0_layers_9_aten__relu_75)
        del layers__9__conv2_input_batch_physical_g0_layers_9_aten__relu_75
        layers__9__conv2_output = layers__9__conv2
        del layers__9__conv2
        layers_9_aten__relu_76_input_batch_physical_g0_layers__9__conv2 = layers__9__conv2_output
        del layers__9__conv2_output
        layers_9_aten__relu_76 = F.relu(layers_9_aten__relu_76_input_batch_physical_g0_layers__9__conv2, )
        del layers_9_aten__relu_76_input_batch_physical_g0_layers__9__conv2
        layers_9_aten__relu_76_output = layers_9_aten__relu_76
        del layers_9_aten__relu_76
        layers__10__conv1_input_batch_physical_g0_layers_9_aten__relu_76 = layers_9_aten__relu_76_output
        layers__10__conv1 = self.layers__10__conv1(layers__10__conv1_input_batch_physical_g0_layers_9_aten__relu_76)
        del layers__10__conv1_input_batch_physical_g0_layers_9_aten__relu_76
        layers__10__conv1_output = layers__10__conv1
        del layers__10__conv1
        layers_10_aten__add_77_input_batch_physical_g0_layers__10__conv1 = layers__10__conv1_output
        del layers__10__conv1_output
        layers__10__parallel_convs_input_batch_physical_g0_layers_9_aten__relu_76 = layers_9_aten__relu_76_output
        del layers_9_aten__relu_76_output
        layers__10__parallel_convs_BatchSizeView_pre = self.layers__10__parallel_convs_BatchSizeView_pre(layers__10__parallel_convs_input_batch_physical_g0_layers_9_aten__relu_76)
        del layers__10__parallel_convs_input_batch_physical_g0_layers_9_aten__relu_76
        layers__10__parallel_convs = self.layers__10__parallel_convs(layers__10__parallel_convs_BatchSizeView_pre)
        del layers__10__parallel_convs_BatchSizeView_pre
        layers__10__parallel_convs_BatchSizeView = self.layers__10__parallel_convs_BatchSizeView(layers__10__parallel_convs)
        del layers__10__parallel_convs
        layers__10__parallel_convs_output = layers__10__parallel_convs_BatchSizeView
        del layers__10__parallel_convs_BatchSizeView
        layers_10_aten__add_77_input_batch_physical_g0_layers__10__parallel_convs = layers__10__parallel_convs_output
        del layers__10__parallel_convs_output
        layers_10_aten__add_77 = layers_10_aten__add_77_input_batch_physical_g0_layers__10__conv1 + layers_10_aten__add_77_input_batch_physical_g0_layers__10__parallel_convs
        del layers_10_aten__add_77_input_batch_physical_g0_layers__10__conv1
        del layers_10_aten__add_77_input_batch_physical_g0_layers__10__parallel_convs
        layers_10_aten__add_77_output = layers_10_aten__add_77
        del layers_10_aten__add_77
        layers_10_aten__relu_78_input_batch_physical_g0_layers_10_aten__add_77 = layers_10_aten__add_77_output
        del layers_10_aten__add_77_output
        layers_10_aten__relu_78 = F.relu(layers_10_aten__relu_78_input_batch_physical_g0_layers_10_aten__add_77, )
        del layers_10_aten__relu_78_input_batch_physical_g0_layers_10_aten__add_77
        layers_10_aten__relu_78_output = layers_10_aten__relu_78
        del layers_10_aten__relu_78
        layers__10__conv2_input_batch_physical_g0_layers_10_aten__relu_78 = layers_10_aten__relu_78_output
        del layers_10_aten__relu_78_output
        layers__10__conv2 = self.layers__10__conv2(layers__10__conv2_input_batch_physical_g0_layers_10_aten__relu_78)
        del layers__10__conv2_input_batch_physical_g0_layers_10_aten__relu_78
        layers__10__conv2_output = layers__10__conv2
        del layers__10__conv2
        layers_10_aten__relu_79_input_batch_physical_g0_layers__10__conv2 = layers__10__conv2_output
        del layers__10__conv2_output
        layers_10_aten__relu_79 = F.relu(layers_10_aten__relu_79_input_batch_physical_g0_layers__10__conv2, )
        del layers_10_aten__relu_79_input_batch_physical_g0_layers__10__conv2
        layers_10_aten__relu_79_output = layers_10_aten__relu_79
        del layers_10_aten__relu_79
        layers__11__conv1_input_batch_physical_g0_layers_10_aten__relu_79 = layers_10_aten__relu_79_output
        layers__11__conv1 = self.layers__11__conv1(layers__11__conv1_input_batch_physical_g0_layers_10_aten__relu_79)
        del layers__11__conv1_input_batch_physical_g0_layers_10_aten__relu_79
        layers__11__conv1_output = layers__11__conv1
        del layers__11__conv1
        layers_11_aten__add_80_input_batch_physical_g0_layers__11__conv1 = layers__11__conv1_output
        del layers__11__conv1_output
        layers__11__parallel_convs_input_batch_physical_g0_layers_10_aten__relu_79 = layers_10_aten__relu_79_output
        del layers_10_aten__relu_79_output
        layers__11__parallel_convs_BatchSizeView_pre = self.layers__11__parallel_convs_BatchSizeView_pre(layers__11__parallel_convs_input_batch_physical_g0_layers_10_aten__relu_79)
        del layers__11__parallel_convs_input_batch_physical_g0_layers_10_aten__relu_79
        layers__11__parallel_convs = self.layers__11__parallel_convs(layers__11__parallel_convs_BatchSizeView_pre)
        del layers__11__parallel_convs_BatchSizeView_pre
        layers__11__parallel_convs_BatchSizeView = self.layers__11__parallel_convs_BatchSizeView(layers__11__parallel_convs)
        del layers__11__parallel_convs
        layers__11__parallel_convs_output = layers__11__parallel_convs_BatchSizeView
        del layers__11__parallel_convs_BatchSizeView
        layers_11_aten__add_80_input_batch_physical_g0_layers__11__parallel_convs = layers__11__parallel_convs_output
        del layers__11__parallel_convs_output
        layers_11_aten__add_80 = layers_11_aten__add_80_input_batch_physical_g0_layers__11__conv1 + layers_11_aten__add_80_input_batch_physical_g0_layers__11__parallel_convs
        del layers_11_aten__add_80_input_batch_physical_g0_layers__11__conv1
        del layers_11_aten__add_80_input_batch_physical_g0_layers__11__parallel_convs
        layers_11_aten__add_80_output = layers_11_aten__add_80
        del layers_11_aten__add_80
        layers_11_aten__relu_81_input_batch_physical_g0_layers_11_aten__add_80 = layers_11_aten__add_80_output
        del layers_11_aten__add_80_output
        layers_11_aten__relu_81 = F.relu(layers_11_aten__relu_81_input_batch_physical_g0_layers_11_aten__add_80, )
        del layers_11_aten__relu_81_input_batch_physical_g0_layers_11_aten__add_80
        layers_11_aten__relu_81_output = layers_11_aten__relu_81
        del layers_11_aten__relu_81
        layers__11__conv2_input_batch_physical_g0_layers_11_aten__relu_81 = layers_11_aten__relu_81_output
        del layers_11_aten__relu_81_output
        layers__11__conv2 = self.layers__11__conv2(layers__11__conv2_input_batch_physical_g0_layers_11_aten__relu_81)
        del layers__11__conv2_input_batch_physical_g0_layers_11_aten__relu_81
        layers__11__conv2_output = layers__11__conv2
        del layers__11__conv2
        layers_11_aten__relu_82_input_batch_physical_g0_layers__11__conv2 = layers__11__conv2_output
        del layers__11__conv2_output
        layers_11_aten__relu_82 = F.relu(layers_11_aten__relu_82_input_batch_physical_g0_layers__11__conv2, )
        del layers_11_aten__relu_82_input_batch_physical_g0_layers__11__conv2
        layers_11_aten__relu_82_output = layers_11_aten__relu_82
        del layers_11_aten__relu_82
        layers__12__conv1_input_batch_physical_g0_layers_11_aten__relu_82 = layers_11_aten__relu_82_output
        layers__12__conv1 = self.layers__12__conv1(layers__12__conv1_input_batch_physical_g0_layers_11_aten__relu_82)
        del layers__12__conv1_input_batch_physical_g0_layers_11_aten__relu_82
        layers__12__conv1_output = layers__12__conv1
        del layers__12__conv1
        layers_12_aten__add_83_input_batch_physical_g0_layers__12__conv1 = layers__12__conv1_output
        del layers__12__conv1_output
        layers__12__parallel_convs_input_batch_physical_g0_layers_11_aten__relu_82 = layers_11_aten__relu_82_output
        del layers_11_aten__relu_82_output
        layers__12__parallel_convs_BatchSizeView_pre = self.layers__12__parallel_convs_BatchSizeView_pre(layers__12__parallel_convs_input_batch_physical_g0_layers_11_aten__relu_82)
        del layers__12__parallel_convs_input_batch_physical_g0_layers_11_aten__relu_82
        layers__12__parallel_convs = self.layers__12__parallel_convs(layers__12__parallel_convs_BatchSizeView_pre)
        del layers__12__parallel_convs_BatchSizeView_pre
        layers__12__parallel_convs_BatchSizeView = self.layers__12__parallel_convs_BatchSizeView(layers__12__parallel_convs)
        del layers__12__parallel_convs
        layers__12__parallel_convs_output = layers__12__parallel_convs_BatchSizeView
        del layers__12__parallel_convs_BatchSizeView
        layers_12_aten__add_83_input_batch_physical_g0_layers__12__parallel_convs = layers__12__parallel_convs_output
        del layers__12__parallel_convs_output
        layers_12_aten__add_83 = layers_12_aten__add_83_input_batch_physical_g0_layers__12__conv1 + layers_12_aten__add_83_input_batch_physical_g0_layers__12__parallel_convs
        del layers_12_aten__add_83_input_batch_physical_g0_layers__12__conv1
        del layers_12_aten__add_83_input_batch_physical_g0_layers__12__parallel_convs
        layers_12_aten__add_83_output = layers_12_aten__add_83
        del layers_12_aten__add_83
        layers_12_aten__relu_84_input_batch_physical_g0_layers_12_aten__add_83 = layers_12_aten__add_83_output
        del layers_12_aten__add_83_output
        layers_12_aten__relu_84 = F.relu(layers_12_aten__relu_84_input_batch_physical_g0_layers_12_aten__add_83, )
        del layers_12_aten__relu_84_input_batch_physical_g0_layers_12_aten__add_83
        layers_12_aten__relu_84_output = layers_12_aten__relu_84
        del layers_12_aten__relu_84
        layers__12__conv2_input_batch_physical_g0_layers_12_aten__relu_84 = layers_12_aten__relu_84_output
        del layers_12_aten__relu_84_output
        layers__12__conv2 = self.layers__12__conv2(layers__12__conv2_input_batch_physical_g0_layers_12_aten__relu_84)
        del layers__12__conv2_input_batch_physical_g0_layers_12_aten__relu_84
        layers__12__conv2_output = layers__12__conv2
        del layers__12__conv2
        layers_12_aten__relu_85_input_batch_physical_g0_layers__12__conv2 = layers__12__conv2_output
        del layers__12__conv2_output
        layers_12_aten__relu_85 = F.relu(layers_12_aten__relu_85_input_batch_physical_g0_layers__12__conv2, )
        del layers_12_aten__relu_85_input_batch_physical_g0_layers__12__conv2
        layers_12_aten__relu_85_output = layers_12_aten__relu_85
        del layers_12_aten__relu_85
        _aten__avg_pool2d_43_input_batch_physical_g0_layers_12_aten__relu_85 = layers_12_aten__relu_85_output
        del layers_12_aten__relu_85_output
        _aten__avg_pool2d_43 = F.avg_pool2d(_aten__avg_pool2d_43_input_batch_physical_g0_layers_12_aten__relu_85, [2, 2], [], [0, 0], False, True)
        del _aten__avg_pool2d_43_input_batch_physical_g0_layers_12_aten__relu_85
        _aten__avg_pool2d_43_output = _aten__avg_pool2d_43
        del _aten__avg_pool2d_43
        _aten__size_44_input_batch_physical_g0__aten__avg_pool2d_43 = _aten__avg_pool2d_43_output
        _aten__size_44 = _aten__size_44_input_batch_physical_g0__aten__avg_pool2d_43.size(0)
        del _aten__size_44_input_batch_physical_g0__aten__avg_pool2d_43
        _aten__size_44_output = _aten__size_44
        del _aten__size_44
        _aten__Int_45_input_batch_physical_g0__aten__size_44 = _aten__size_44_output
        del _aten__size_44_output
        _aten__Int_45 = int(_aten__Int_45_input_batch_physical_g0__aten__size_44)
        del _aten__Int_45_input_batch_physical_g0__aten__size_44
        _aten__Int_45_output = _aten__Int_45
        del _aten__Int_45
        _aten__view_46_input_batch_physical_g0__aten__Int_45 = _aten__Int_45_output
        del _aten__Int_45_output
        _aten__view_46_input_batch_physical_g0__aten__avg_pool2d_43 = _aten__avg_pool2d_43_output
        del _aten__avg_pool2d_43_output
        _aten__view_46 = _aten__view_46_input_batch_physical_g0__aten__avg_pool2d_43.view(_aten__view_46_input_batch_physical_g0__aten__Int_45, -1)
        del _aten__view_46_input_batch_physical_g0__aten__avg_pool2d_43
        del _aten__view_46_input_batch_physical_g0__aten__Int_45
        _aten__view_46_output = _aten__view_46
        del _aten__view_46
        linear_input_batch_physical_g0__aten__view_46 = _aten__view_46_output
        del _aten__view_46_output
        linear = self.linear(linear_input_batch_physical_g0__aten__view_46)
        del linear_input_batch_physical_g0__aten__view_46
        linear_output = linear
        del linear
        origin_x_input_dedup_physical_g0_input_1 = input_1
        expand_dedup_physical_g0_input_1 = self.expand_dedup_physical_g0_input_1(origin_x_input_dedup_physical_g0_input_1)
        del origin_x_input_dedup_physical_g0_input_1
        origin_x = expand_dedup_physical_g0_input_1
        del expand_dedup_physical_g0_input_1
        origin_y_input_dedup_physical_g0_input_2 = input_2
        expand_dedup_physical_g0_input_2 = self.expand_dedup_physical_g0_input_2(origin_y_input_dedup_physical_g0_input_2)
        del origin_y_input_dedup_physical_g0_input_2
        origin_y = expand_dedup_physical_g0_input_2
        del expand_dedup_physical_g0_input_2
        output_2294_input_batch_physical_g0_linear = linear_output
        del linear_output
        output_2294 = output_2294_input_batch_physical_g0_linear
        del output_2294_input_batch_physical_g0_linear
        origin_x_output = origin_x
        origin_y_output = origin_y
        output_2294_output = output_2294
        return origin_x_output, origin_y_output, output_2294_output