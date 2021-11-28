import functools
import math
import random

import torch
import torch.nn as nn
import tqdm
from mmcv.utils.logging import print_log



from common.searchspace import BiasedMixedOp, MixedOp, SearchSpace
from configs.searchspace import ProxylessConfig, ProxylessStageConfig
from .utils import ConvBNReLU, InvertedResidual, make_divisible


from typing import Callable, Optional, List

import torch
import torch.nn as nn


def shuffle_layer(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    # transpose
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(
        kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2


def make_divisible(v, divisor, min_val=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def tf_indices_to_pytorch_spec(tf_indices, pytorch_space):
    tf_indices = list(map(int, tf_indices.split(':')))
    if len(tf_indices) == 22:
        assert len(tf_indices) == len(pytorch_space)
        return {k: v[i] if isinstance(v, list) else v[0][i] for i, (k, v) in zip(tf_indices, pytorch_space.items())}

    indices = [3, 6, 9, 12, 16, 19, 22, 25, 29, 32, 35, 38, 42, 45, 48, 51, 55, 58, 61, 64, 68]
    assert len(indices) == 21
    assert len(pytorch_space) == 22
    result = {}
    for i, key in zip([None] + indices, pytorch_space.keys()):
        if i is None:
            assert len(pytorch_space[key]) == 1
            chosen = pytorch_space[key][0]
        else:
            kernel_size = [3, 5, 7][tf_indices[i]]
            expand_ratio = [3, 6][tf_indices[i + 1]]
            skip = tf_indices[i + 2]
            if skip:
                chosen = 'skip'
            else:
                chosen = f'k{kernel_size}e{expand_ratio}'
        assert chosen in pytorch_space[key] or chosen in pytorch_space[key][0], f'{i}, {chosen}, {pytorch_space[key]}'
        result[key] = chosen
    return result


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                      bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )
        self.out_channels = out_planes


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        kernel_size: int = 3,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, kernel_size=kernel_size, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class _MbNet(nn.Module):
    def __init__(self, first_conv, blocks, feature_mix_layer, dropout_layer, classifier):
        super().__init__()
        self.first_conv = first_conv
        self.blocks = nn.Sequential(*blocks)
        self.feature_mix_layer = feature_mix_layer
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout_layer = dropout_layer
        self.classifier = classifier

    def forward(self, x):
        x = self.first_conv(x)
        x = self.blocks(x)
        x = self.feature_mix_layer(x)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.dropout_layer(x)
        x = self.classifier(x)
        return x

    def no_weight_decay(self):
        # no regularizer to linear layer
        return {'classifier.weight', 'classifier.bias'}

    def reset_parameters(self, model_init='he_fout', init_div_groups=False,
                         bn_momentum=0.1, bn_eps=1e-5,
                         track_running_stats=True, zero_grad=False):
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
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                m.momentum = bn_momentum
                m.eps = bn_eps
                if not track_running_stats and m.track_running_stats:
                    m.track_running_stats = False
                    delattr(m, 'running_mean')
                    delattr(m, 'running_var')
                    delattr(m, 'num_batches_tracked')
                    m.register_parameter('running_mean', None)
                    m.register_parameter('running_var', None)
                    m.register_parameter('num_batches_tracked', None)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # zero out gradients
        if zero_grad:
            for p in self.parameters():
                p.grad = torch.zeros_like(p)


class _MbMixLayer(nn.Module):
    def __init__(self, ops, **metainfo):
        super().__init__()
        for name, op in ops.items():
            self.add_module(name, op)
        self.ops = list(ops.keys())
        self.metainfo = metainfo
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.world_rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        else:
            self.world_rank = 0
            self.world_size = 1

        self.fixed = None

    def _sample(self):
        chosen = None
        for i in range(self.world_size):
            tmp = random.choice(self.ops)
            if i == self.world_rank:
                chosen = tmp
        assert chosen is not None
        return chosen

    def forward(self, x):
        if self.fixed is not None:
            return getattr(self, self.fixed)(x)
        return getattr(self, self._sample())(x)

    def summary(self):
        return 'MbMixLayer(' + ', '.join([f'{k}={v}' for k, v in {'ops': self.ops, **self.metainfo}.items()]) + ')'


class ProxylessNAS(_MbNet, SearchSpace):
    def __init__(self, config: ProxylessConfig, reset_parameters=True):
        stem_width = make_divisible(config.width_mult * config.stem_width, 8)

        first_conv = ConvBNReLU(3, stem_width, stride=2, norm_layer=nn.BatchNorm2d)

        last_width = stem_width
        blocks = []
        for i, stage_config in enumerate(config.stages, start=1):
            print_log(f'Building stage #{i}...', __name__)
            width = make_divisible(stage_config.width * config.width_mult, 8)
            blocks += self._build_stage(i, stage_config, last_width, width)
            last_width = width

        final_width = make_divisible(1280 * config.width_mult, 8) if config.width_mult > 1 else 1280
        dropout_layer = nn.Dropout(config.dropout_rate)
        feature_mix_layer = ConvBNReLU(last_width, final_width, kernel_size=1, norm_layer=nn.BatchNorm2d)
        classifier = nn.Linear(final_width, config.num_labels)
        super().__init__(first_conv, blocks, feature_mix_layer, dropout_layer, classifier)

        if reset_parameters:
            self.reset_parameters(track_running_stats=False, zero_grad=True)

    def _build_stage(self, stage_idx: int, config: ProxylessStageConfig, input_width: int, output_width: int):
        depth_min, depth_max = config.depth_range
        blocks = []
        for i in range(depth_max):
            stride = 2 if config.downsample and i == 0 else 1
            op_choices = {}
            for exp_ratio in config.exp_ratio_range:
                for kernel_size in config.kernel_size_range:
                    op_choices[f'k{kernel_size}e{exp_ratio}'] = InvertedResidual(input_width, output_width, stride, exp_ratio, kernel_size)
            if i >= depth_min:
                prior = [0.5 / len(op_choices)] * len(op_choices) + [0.5]
                op_choices['skip'] = nn.Identity()
                blocks.append(BiasedMixedOp(f's{stage_idx}b{i + 1}_i{input_width}o{output_width}', op_choices, prior))
                assert blocks[-1].op_candidates[-1] == 'skip'
            else:
                blocks.append(MixedOp(f's{stage_idx}b{i + 1}_i{input_width}o{output_width}', op_choices))
            print_log(f'Created block: {blocks[-1].key}: {blocks[-1].op_candidates}', __name__)
            input_width = output_width
        return blocks

    def reset_running_stats(self, dataloader, max_steps=200):
        bn_mean = {}
        bn_var = {}

        def bn_forward_hook(bn, inputs, outputs, mean_est, var_est):
            aggregate_dimensions = (0, 2, 3)
            inputs = inputs[0]  # input is a tuple of arguments
            batch_mean = inputs.mean(aggregate_dimensions, keepdim=True)  # 1, C, 1, 1
            batch_var = (inputs - batch_mean) ** 2
            batch_var = batch_var.mean(aggregate_dimensions, keepdim=True)

            batch_mean = torch.squeeze(batch_mean)
            batch_var = torch.squeeze(batch_var)

            mean_est.append(batch_mean.data)
            var_est.append(batch_var.data)

        handles = []
        for name, m in self.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                bn_mean[name] = []
                bn_var[name] = []
                handle = m.register_forward_hook(functools.partial(bn_forward_hook, mean_est=bn_mean[name], var_est=bn_var[name]))
                handles.append(handle)

        self.train()
        with torch.no_grad():
            pbar = tqdm.tqdm(range(max_steps), desc='Calibrating BatchNorm')
            for _ in pbar:
                images, _ = next(dataloader)
                self(images)

            for name, m in self.named_modules():
                if name in bn_mean and len(bn_mean[name]) > 0:
                    feature_dim = bn_mean[name][0].size(0)
                    assert isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
                    m.running_mean.data[:feature_dim].copy_(sum(bn_mean[name]) / len(bn_mean[name]))
                    m.running_var.data[:feature_dim].copy_(sum(bn_var[name]) / len(bn_var[name]))

        for handle in handles:
            handle.remove()

    def fix_sample(self, sample):
        if isinstance(sample, list):
            search_space = self.export_search_space()
            assert len(search_space) == len(sample)
            sample = {k: v for k, v in zip(search_space.keys(), sample)}
        else:
            assert len(self.export_search_space()) == len(sample)
        for name, module in self.named_modules():
            if isinstance(module, _MbMixLayer) and name in sample:
                module.fixed = sample[name]
        return sample

    def export_search_space(self):
        result = {}
        for name, module in self.named_modules():
            if isinstance(module, _MbMixLayer) and len(module.ops) > 1:
                result[name] = module.ops
        return result