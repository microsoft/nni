import torch
import torch.nn as nn
import torch.nn.functional as F

import sdk.custom_ops_torch as CUSTOM
from nni.nas.pytorch import mutables





class Graph(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers__0__conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=32, padding_mode='zeros')
        self.layers__0__parallel_convs = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, padding_mode='zeros')
        self.layers__0__conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, padding_mode='zeros')
        self.layers__1__conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1), groups=64, padding_mode='zeros')
        self.layers__1__parallel_convs = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0), dilation=(1, 1), groups=1, padding_mode='zeros')
        self.layers__1__conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, padding_mode='zeros')
        self.layers__2__conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=128, padding_mode='zeros')
        self.layers__2__parallel_convs = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, padding_mode='zeros')
        self.layers__2__conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, padding_mode='zeros')
        self.layers__3__conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1), groups=128, padding_mode='zeros')
        self.layers__3__parallel_convs = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0), dilation=(1, 1), groups=1, padding_mode='zeros')
        self.layers__3__conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, padding_mode='zeros')
        self.layers__4__conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=256, padding_mode='zeros')
        self.layers__4__parallel_convs = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, padding_mode='zeros')
        self.layers__4__conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, padding_mode='zeros')
        self.layers__5__conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1), groups=256, padding_mode='zeros')
        self.layers__5__parallel_convs = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0), dilation=(1, 1), groups=1, padding_mode='zeros')
        self.layers__5__conv2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, padding_mode='zeros')
        self.layers__6__conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=512, padding_mode='zeros')
        self.layers__6__parallel_convs = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, padding_mode='zeros')
        self.layers__6__conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, padding_mode='zeros')
        self.layers__7__conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=512, padding_mode='zeros')
        self.layers__7__parallel_convs = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, padding_mode='zeros')
        self.layers__7__conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, padding_mode='zeros')
        self.layers__8__conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=512, padding_mode='zeros')
        self.layers__8__parallel_convs = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, padding_mode='zeros')
        self.layers__8__conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, padding_mode='zeros')
        self.layers__9__conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=512, padding_mode='zeros')
        self.layers__9__parallel_convs = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, padding_mode='zeros')
        self.layers__9__conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, padding_mode='zeros')
        self.layers__10__conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=512, padding_mode='zeros')
        self.layers__10__parallel_convs = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, padding_mode='zeros')
        self.layers__10__conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, padding_mode='zeros')
        self.layers__11__conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1), groups=512, padding_mode='zeros')
        self.layers__11__parallel_convs = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0), dilation=(1, 1), groups=1, padding_mode='zeros')
        self.layers__11__conv2 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, padding_mode='zeros')
        self.layers__12__conv1 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1024, padding_mode='zeros')
        self.layers__12__parallel_convs = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, padding_mode='zeros')
        self.layers__12__conv2 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, padding_mode='zeros')
        self.linear = nn.Linear(in_features=1024, out_features=10, bias=True)

    def forward(self, input_1, input_2):
        input_1 = input_1.cuda()
        input_2 = input_2.cuda()
        conv1 = self.conv1(input_1)
        bn1 = self.bn1(conv1)
        del conv1
        _aten__relu_42 = F.relu(bn1, )
        del bn1
        layers__0__conv1 = self.layers__0__conv1(_aten__relu_42)
        layers__0__parallel_convs = self.layers__0__parallel_convs(_aten__relu_42)
        del _aten__relu_42
        layers_0_aten__add_47 = layers__0__conv1 + layers__0__parallel_convs
        del layers__0__conv1
        del layers__0__parallel_convs
        layers_0_aten__relu_48 = F.relu(layers_0_aten__add_47, )
        del layers_0_aten__add_47
        layers__0__conv2 = self.layers__0__conv2(layers_0_aten__relu_48)
        del layers_0_aten__relu_48
        layers_0_aten__relu_49 = F.relu(layers__0__conv2, )
        del layers__0__conv2
        layers__1__conv1 = self.layers__1__conv1(layers_0_aten__relu_49)
        layers__1__parallel_convs = self.layers__1__parallel_convs(layers_0_aten__relu_49)
        del layers_0_aten__relu_49
        layers_1_aten__add_50 = layers__1__conv1 + layers__1__parallel_convs
        del layers__1__conv1
        del layers__1__parallel_convs
        layers_1_aten__relu_51 = F.relu(layers_1_aten__add_50, )
        del layers_1_aten__add_50
        layers__1__conv2 = self.layers__1__conv2(layers_1_aten__relu_51)
        del layers_1_aten__relu_51
        layers_1_aten__relu_52 = F.relu(layers__1__conv2, )
        del layers__1__conv2
        layers__2__conv1 = self.layers__2__conv1(layers_1_aten__relu_52)
        layers__2__parallel_convs = self.layers__2__parallel_convs(layers_1_aten__relu_52)
        del layers_1_aten__relu_52
        layers_2_aten__add_53 = layers__2__conv1 + layers__2__parallel_convs
        del layers__2__conv1
        del layers__2__parallel_convs
        layers_2_aten__relu_54 = F.relu(layers_2_aten__add_53, )
        del layers_2_aten__add_53
        layers__2__conv2 = self.layers__2__conv2(layers_2_aten__relu_54)
        del layers_2_aten__relu_54
        layers_2_aten__relu_55 = F.relu(layers__2__conv2, )
        del layers__2__conv2
        layers__3__conv1 = self.layers__3__conv1(layers_2_aten__relu_55)
        layers__3__parallel_convs = self.layers__3__parallel_convs(layers_2_aten__relu_55)
        del layers_2_aten__relu_55
        layers_3_aten__add_56 = layers__3__conv1 + layers__3__parallel_convs
        del layers__3__conv1
        del layers__3__parallel_convs
        layers_3_aten__relu_57 = F.relu(layers_3_aten__add_56, )
        del layers_3_aten__add_56
        layers__3__conv2 = self.layers__3__conv2(layers_3_aten__relu_57)
        del layers_3_aten__relu_57
        layers_3_aten__relu_58 = F.relu(layers__3__conv2, )
        del layers__3__conv2
        layers__4__conv1 = self.layers__4__conv1(layers_3_aten__relu_58)
        layers__4__parallel_convs = self.layers__4__parallel_convs(layers_3_aten__relu_58)
        del layers_3_aten__relu_58
        layers_4_aten__add_59 = layers__4__conv1 + layers__4__parallel_convs
        del layers__4__conv1
        del layers__4__parallel_convs
        layers_4_aten__relu_60 = F.relu(layers_4_aten__add_59, )
        del layers_4_aten__add_59
        layers__4__conv2 = self.layers__4__conv2(layers_4_aten__relu_60)
        del layers_4_aten__relu_60
        layers_4_aten__relu_61 = F.relu(layers__4__conv2, )
        del layers__4__conv2
        layers__5__conv1 = self.layers__5__conv1(layers_4_aten__relu_61)
        layers__5__parallel_convs = self.layers__5__parallel_convs(layers_4_aten__relu_61)
        del layers_4_aten__relu_61
        layers_5_aten__add_62 = layers__5__conv1 + layers__5__parallel_convs
        del layers__5__conv1
        del layers__5__parallel_convs
        layers_5_aten__relu_63 = F.relu(layers_5_aten__add_62, )
        del layers_5_aten__add_62
        layers__5__conv2 = self.layers__5__conv2(layers_5_aten__relu_63)
        del layers_5_aten__relu_63
        layers_5_aten__relu_64 = F.relu(layers__5__conv2, )
        del layers__5__conv2
        layers__6__conv1 = self.layers__6__conv1(layers_5_aten__relu_64)
        layers__6__parallel_convs = self.layers__6__parallel_convs(layers_5_aten__relu_64)
        del layers_5_aten__relu_64
        layers_6_aten__add_65 = layers__6__conv1 + layers__6__parallel_convs
        del layers__6__conv1
        del layers__6__parallel_convs
        layers_6_aten__relu_66 = F.relu(layers_6_aten__add_65, )
        del layers_6_aten__add_65
        layers__6__conv2 = self.layers__6__conv2(layers_6_aten__relu_66)
        del layers_6_aten__relu_66
        layers_6_aten__relu_67 = F.relu(layers__6__conv2, )
        del layers__6__conv2
        layers__7__conv1 = self.layers__7__conv1(layers_6_aten__relu_67)
        layers__7__parallel_convs = self.layers__7__parallel_convs(layers_6_aten__relu_67)
        del layers_6_aten__relu_67
        layers_7_aten__add_68 = layers__7__conv1 + layers__7__parallel_convs
        del layers__7__conv1
        del layers__7__parallel_convs
        layers_7_aten__relu_69 = F.relu(layers_7_aten__add_68, )
        del layers_7_aten__add_68
        layers__7__conv2 = self.layers__7__conv2(layers_7_aten__relu_69)
        del layers_7_aten__relu_69
        layers_7_aten__relu_70 = F.relu(layers__7__conv2, )
        del layers__7__conv2
        layers__8__conv1 = self.layers__8__conv1(layers_7_aten__relu_70)
        layers__8__parallel_convs = self.layers__8__parallel_convs(layers_7_aten__relu_70)
        del layers_7_aten__relu_70
        layers_8_aten__add_71 = layers__8__conv1 + layers__8__parallel_convs
        del layers__8__conv1
        del layers__8__parallel_convs
        layers_8_aten__relu_72 = F.relu(layers_8_aten__add_71, )
        del layers_8_aten__add_71
        layers__8__conv2 = self.layers__8__conv2(layers_8_aten__relu_72)
        del layers_8_aten__relu_72
        layers_8_aten__relu_73 = F.relu(layers__8__conv2, )
        del layers__8__conv2
        layers__9__conv1 = self.layers__9__conv1(layers_8_aten__relu_73)
        layers__9__parallel_convs = self.layers__9__parallel_convs(layers_8_aten__relu_73)
        del layers_8_aten__relu_73
        layers_9_aten__add_74 = layers__9__conv1 + layers__9__parallel_convs
        del layers__9__conv1
        del layers__9__parallel_convs
        layers_9_aten__relu_75 = F.relu(layers_9_aten__add_74, )
        del layers_9_aten__add_74
        layers__9__conv2 = self.layers__9__conv2(layers_9_aten__relu_75)
        del layers_9_aten__relu_75
        layers_9_aten__relu_76 = F.relu(layers__9__conv2, )
        del layers__9__conv2
        layers__10__conv1 = self.layers__10__conv1(layers_9_aten__relu_76)
        layers__10__parallel_convs = self.layers__10__parallel_convs(layers_9_aten__relu_76)
        del layers_9_aten__relu_76
        layers_10_aten__add_77 = layers__10__conv1 + layers__10__parallel_convs
        del layers__10__conv1
        del layers__10__parallel_convs
        layers_10_aten__relu_78 = F.relu(layers_10_aten__add_77, )
        del layers_10_aten__add_77
        layers__10__conv2 = self.layers__10__conv2(layers_10_aten__relu_78)
        del layers_10_aten__relu_78
        layers_10_aten__relu_79 = F.relu(layers__10__conv2, )
        del layers__10__conv2
        layers__11__conv1 = self.layers__11__conv1(layers_10_aten__relu_79)
        layers__11__parallel_convs = self.layers__11__parallel_convs(layers_10_aten__relu_79)
        del layers_10_aten__relu_79
        layers_11_aten__add_80 = layers__11__conv1 + layers__11__parallel_convs
        del layers__11__conv1
        del layers__11__parallel_convs
        layers_11_aten__relu_81 = F.relu(layers_11_aten__add_80, )
        del layers_11_aten__add_80
        layers__11__conv2 = self.layers__11__conv2(layers_11_aten__relu_81)
        del layers_11_aten__relu_81
        layers_11_aten__relu_82 = F.relu(layers__11__conv2, )
        del layers__11__conv2
        layers__12__conv1 = self.layers__12__conv1(layers_11_aten__relu_82)
        layers__12__parallel_convs = self.layers__12__parallel_convs(layers_11_aten__relu_82)
        del layers_11_aten__relu_82
        layers_12_aten__add_83 = layers__12__conv1 + layers__12__parallel_convs
        del layers__12__conv1
        del layers__12__parallel_convs
        layers_12_aten__relu_84 = F.relu(layers_12_aten__add_83, )
        del layers_12_aten__add_83
        layers__12__conv2 = self.layers__12__conv2(layers_12_aten__relu_84)
        del layers_12_aten__relu_84
        layers_12_aten__relu_85 = F.relu(layers__12__conv2, )
        del layers__12__conv2
        _aten__avg_pool2d_43 = F.avg_pool2d(layers_12_aten__relu_85, [2, 2], [], [0, 0], False, True)
        del layers_12_aten__relu_85
        _aten__size_44 = _aten__avg_pool2d_43.size(0)
        _aten__Int_45 = int(_aten__size_44)
        del _aten__size_44
        _aten__view_46 = _aten__avg_pool2d_43.view(_aten__Int_45, -1)
        del _aten__avg_pool2d_43
        del _aten__Int_45
        linear = self.linear(_aten__view_46)
        del _aten__view_46
        origin_x = input_1
        origin_y = input_2
        output_2294 = linear
        return origin_x, origin_y, output_2294