import torch
import torch.nn as nn
from torch.nn import functional as F

from nni.nas.pytorch import mutables
from models.hbuilder import *

DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }

_DEBUG = False


class SuperNet(nn.Module):

    def __init__(self, block_args, choices, num_classes=1000, in_chans=3, stem_size=16, num_features=1280,
                 head_bias=True,
                 channel_multiplier=1.0, pad_type='', act_layer=nn.ReLU, drop_rate=0., drop_path_rate=0., slice=4,
                 se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None, global_pool='avg', resunit=False,
                 dil_conv=False):
        super(SuperNet, self).__init__()

        self.num_classes = num_classes
        self.num_features = num_features
        self.drop_rate = drop_rate
        self._in_chs = in_chans

        # Stem
        stem_size = round_channels(stem_size, channel_multiplier)
        self.conv_stem = create_conv2d(self._in_chs, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        self._in_chs = stem_size

        # Middle stages (IR/ER/DS Blocks)
        builder = SuperNetBuilder(
            choices, channel_multiplier, 8, None, 32, pad_type, act_layer, se_kwargs,
            norm_layer, norm_kwargs, drop_path_rate, verbose=_DEBUG, resunit=resunit, dil_conv=dil_conv)
        # self.blocks = nn.ModuleList(*builder(self._in_chs, block_args))
        blocks = builder(self._in_chs, block_args)
        self.blocks = nn.Sequential(*blocks)
        self._in_chs = builder.in_chs

        # Head + Pooling
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.conv_head = create_conv2d(self._in_chs, self.num_features, 1, padding=pad_type, bias=head_bias)
        self.act2 = act_layer(inplace=True)

        # Classifier
        self.classifier = nn.Linear(self.num_features * self.global_pool.feat_mult(), self.num_classes)

        self.meta_layer = nn.Linear(self.num_classes * slice, 1)
        efficientnet_init_weights(self)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.num_classes = num_classes
        self.classifier = nn.Linear(
            self.num_features * self.global_pool.feat_mult(), num_classes) if self.num_classes else None

    def forward_features(self, x, cand):
        # architecture = [[0], [], [], [], [], [0]]
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        if cand is not None:
            pass # x = self.blocks(x)
        else:
            x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        return x

    def forward(self, x, cand=None):
        x = self.forward_features(x, cand)
        x = x.flatten(1)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return self.classifier(x)

    def forward_meta(self, features):
        return self.meta_layer(features.view(1, -1))

    def rand_parameters(self, architecture, meta=False):
        for name, param in self.named_parameters(recurse=True):
            if 'meta' in name and meta:
                yield param
            elif 'blocks' not in name and 'meta' not in name and (not meta):
                yield param

        if not meta:
            for layer, layer_arch in zip(self.blocks, architecture.keys()):
                for choice_idx, choice in enumerate(architecture[layer_arch]):
                    if choice:
                        for name, param in layer[choice_idx].named_parameters(recurse=True):
                            yield param


def search_for_layer(flops_op_dict, arch_def, flops_minimum, flops_maximum):
    sta_num = [1, 1, 1, 1, 1]
    order = [2, 3, 4, 1, 0, 2, 3, 4, 1, 0]
    limits = [3, 3, 3, 2, 2, 4, 4, 4, 4, 4]
    size_factor = 7
    base_min_flops = sum([flops_op_dict[i][0][0] for i in range(5)])
    base_max_flops = sum([flops_op_dict[i][5][0] for i in range(5)])


    if base_min_flops > flops_maximum:
        while base_min_flops > flops_maximum and size_factor >= 2:
            size_factor = size_factor - 1
            flops_minimum = flops_minimum * (7. / size_factor)
            flops_maximum = flops_maximum * (7. / size_factor)
        if size_factor < 2:
            return None, None, None
    elif base_max_flops < flops_minimum:
        cur_ptr = 0
        while base_max_flops < flops_minimum and cur_ptr <= 9:
            if sta_num[order[cur_ptr]] >= limits[cur_ptr]:
                cur_ptr += 1
                continue
            base_max_flops = base_max_flops + flops_op_dict[order[cur_ptr]][5][1]
            sta_num[order[cur_ptr]] += 1
        if cur_ptr > 7 and base_max_flops < flops_minimum:
            return None, None, None

    cur_ptr = 0
    while cur_ptr <= 9:
        if sta_num[order[cur_ptr]] >= limits[cur_ptr]:
            cur_ptr += 1
            continue
        base_max_flops = base_max_flops + flops_op_dict[order[cur_ptr]][5][1]
        if base_max_flops <= flops_maximum:
            sta_num[order[cur_ptr]] += 1
        else:
            break

    arch_def = [item[:i] for i, item in zip([1]+sta_num+[1], arch_def)]
    # print(arch_def)

    return sta_num, arch_def, size_factor

def _gen_supernet(flops_minimum=0, flops_maximum=600, **kwargs):
    choices = {'kernel_size': [3, 5, 7], 'exp_ratio': [4, 6]}

    num_features = 1280

    # act_layer = HardSwish
    act_layer = Swish
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c16_se0.25'],
        # stage 1, 112x112 in
        ['ir_r1_k3_s2_e4_c24_se0.25', 'ir_r1_k3_s1_e4_c24_se0.25', 'ir_r1_k3_s1_e4_c24_se0.25',
         'ir_r1_k3_s1_e4_c24_se0.25'],
        # stage 2, 56x56 in
        ['ir_r1_k5_s2_e4_c40_se0.25', 'ir_r1_k5_s1_e4_c40_se0.25', 'ir_r1_k5_s2_e4_c40_se0.25',
         'ir_r1_k5_s2_e4_c40_se0.25'],
        # stage 3, 28x28 in
        ['ir_r1_k3_s2_e6_c80_se0.25', 'ir_r1_k3_s1_e4_c80_se0.25', 'ir_r1_k3_s1_e4_c80_se0.25',
         'ir_r2_k3_s1_e4_c80_se0.25'],
        # stage 4, 14x14in
        ['ir_r1_k3_s1_e6_c96_se0.25', 'ir_r1_k3_s1_e6_c96_se0.25', 'ir_r1_k3_s1_e6_c96_se0.25',
         'ir_r1_k3_s1_e6_c96_se0.25'],
        # stage 5, 14x14in
        ['ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s1_e6_c192_se0.25', 'ir_r1_k5_s2_e6_c192_se0.25',
         'ir_r1_k5_s2_e6_c192_se0.25'],
        # stage 6, 7x7 in
        ['cn_r1_k1_s1_c320_se0.25'],
    ]

    flops_op_dict = {}
    for i in range(5):
        flops_op_dict[i] = {}
    flops_op_dict[0][0] = (21.828704, 18.820752)
    flops_op_dict[0][1] = (32.669328, 28.16048)
    flops_op_dict[0][2] = (25.039968, 23.637648)
    flops_op_dict[0][3] = (37.486224, 35.385824)
    flops_op_dict[0][4] = (29.856864, 30.862992)
    flops_op_dict[0][5] = (44.711568, 46.22384)
    flops_op_dict[1][0] = (11.808656, 11.86712)
    flops_op_dict[1][1] = (17.68624, 17.780848)
    flops_op_dict[1][2] = (13.01288, 13.87416)
    flops_op_dict[1][3] = (19.492576, 20.791408)
    flops_op_dict[1][4] = (14.819216, 16.88472)
    flops_op_dict[1][5] = (22.20208, 25.307248)
    flops_op_dict[2][0] = (8.198, 10.99632)
    flops_op_dict[2][1] = (12.292848, 16.5172)
    flops_op_dict[2][2] = (8.69976, 11.99984)
    flops_op_dict[2][3] = (13.045488, 18.02248)
    flops_op_dict[2][4] = (9.4524, 13.50512)
    flops_op_dict[2][5] = (14.174448, 20.2804)
    flops_op_dict[3][0] = (12.006112, 15.61632)
    flops_op_dict[3][1] = (18.028752, 23.46096)
    flops_op_dict[3][2] = (13.009632, 16.820544)
    flops_op_dict[3][3] = (19.534032, 25.267296)
    flops_op_dict[3][4] = (14.514912, 18.62688)
    flops_op_dict[3][5] = (21.791952, 27.9768)
    flops_op_dict[4][0] = (11.307456, 15.292416)
    flops_op_dict[4][1] = (17.007072, 23.1504)
    flops_op_dict[4][2] = (11.608512, 15.894528)
    flops_op_dict[4][3] = (17.458656, 24.053568)
    flops_op_dict[4][4] = (12.060096, 16.797696)
    flops_op_dict[4][5] = (18.136032, 25.40832)

    sta_num, arch_def, size_factor = search_for_layer(flops_op_dict, arch_def, flops_minimum, flops_maximum)

    if sta_num is None or arch_def is None or size_factor is None:
        raise ValueError('Invalid FLOPs Settings')

    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        choices=choices,
        num_features=num_features,
        stem_size=16,
        # channel_multiplier=channel_multiplier,
        norm_kwargs=resolve_bn_args(kwargs),
        act_layer=act_layer,
        se_kwargs=dict(act_layer=nn.ReLU, gate_fn=hard_sigmoid, reduce_mid=True, divisor=8),
        **kwargs,
    )
    model = SuperNet(**model_kwargs)
    return model, sta_num, size_factor


class Classifier(nn.Module):
    def __init__(self, num_classes=1000):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        return self.classifier(x)


if __name__ == '__main__':
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c16_se0.25'],
        # stage 1, 112x112 in
        ['ir_r1_k3_s2_e4_c24_se0.25', 'ir_r1_k3_s1_e4_c24_se0.25', 'ir_r1_k3_s1_e4_c24_se0.25',
         'ir_r1_k3_s1_e4_c24_se0.25'],
        # stage 2, 56x56 in
        ['ir_r1_k5_s2_e4_c40_se0.25', 'ir_r1_k5_s1_e4_c40_se0.25', 'ir_r1_k5_s2_e4_c40_se0.25',
         'ir_r1_k5_s2_e4_c40_se0.25'],
        # stage 3, 28x28 in
        ['ir_r1_k3_s2_e6_c80_se0.25', 'ir_r1_k3_s1_e4_c80_se0.25', 'ir_r1_k3_s1_e4_c80_se0.25',
         'ir_r2_k3_s1_e4_c80_se0.25'],
        # stage 4, 14x14in
        ['ir_r1_k3_s1_e6_c96_se0.25', 'ir_r1_k3_s1_e6_c96_se0.25', 'ir_r1_k3_s1_e6_c96_se0.25',
         'ir_r1_k3_s1_e6_c96_se0.25'],
        # stage 5, 14x14in
        ['ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s1_e6_c192_se0.25', 'ir_r1_k5_s2_e6_c192_se0.25',
         'ir_r1_k5_s2_e6_c192_se0.25'],
        # stage 6, 7x7 in
        ['cn_r1_k1_s1_c320_se0.25'],
    ]

    flops_op_dict = {}
    for i in range(5):
        flops_op_dict[i] = {}
    flops_op_dict[0][0] = (21.828704, 18.820752)
    flops_op_dict[0][1] = (32.669328, 28.16048)
    flops_op_dict[0][2] = (25.039968, 23.637648)
    flops_op_dict[0][3] = (37.486224, 35.385824)
    flops_op_dict[0][4] = (29.856864, 30.862992)
    flops_op_dict[0][5] = (44.711568, 46.22384)
    flops_op_dict[1][0] = (11.808656, 11.86712)
    flops_op_dict[1][1] = (17.68624, 17.780848)
    flops_op_dict[1][2] = (13.01288, 13.87416)
    flops_op_dict[1][3] = (19.492576, 20.791408)
    flops_op_dict[1][4] = (14.819216, 16.88472)
    flops_op_dict[1][5] = (22.20208, 25.307248)
    flops_op_dict[2][0] = (8.198, 10.99632)
    flops_op_dict[2][1] = (12.292848, 16.5172)
    flops_op_dict[2][2] = (8.69976, 11.99984)
    flops_op_dict[2][3] = (13.045488, 18.02248)
    flops_op_dict[2][4] = (9.4524, 13.50512)
    flops_op_dict[2][5] = (14.174448, 20.2804)
    flops_op_dict[3][0] = (12.006112, 15.61632)
    flops_op_dict[3][1] = (18.028752, 23.46096)
    flops_op_dict[3][2] = (13.009632, 16.820544)
    flops_op_dict[3][3] = (19.534032, 25.267296)
    flops_op_dict[3][4] = (14.514912, 18.62688)
    flops_op_dict[3][5] = (21.791952, 27.9768)
    flops_op_dict[4][0] = (11.307456, 15.292416)
    flops_op_dict[4][1] = (17.007072, 23.1504)
    flops_op_dict[4][2] = (11.608512, 15.894528)
    flops_op_dict[4][3] = (17.458656, 24.053568)
    flops_op_dict[4][4] = (12.060096, 16.797696)
    flops_op_dict[4][5] = (18.136032, 25.40832)

    sta_num, arch_def, size_factor = search_for_layer(flops_op_dict, arch_def, 0, 20)
    print(sta_num, size_factor)
