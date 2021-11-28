import torch
import torch.nn as nn
import torch.nn.functional as F

from common.dynamic_ops import DynamicBatchNorm2d, DynamicConv2d


OPS = {
    'avg_pool_3x3': lambda C: Pool('avg', C, 3, 1),
    'max_pool_2x2': lambda C: Pool('max', C, 2, 0),
    'max_pool_3x3': lambda C: Pool('max', C, 3, 1),
    'max_pool_5x5': lambda C: Pool('max', C, 5, 2),
    'max_pool_7x7': lambda C: Pool('max', C, 7, 3),
    'skip_connect': lambda C: SkipConnection(C, C),
    'sep_conv_3x3': lambda C: StackedSepConv(C, C, 3, 1),
    'sep_conv_5x5': lambda C: StackedSepConv(C, C, 5, 2),
    'sep_conv_7x7': lambda C: StackedSepConv(C, C, 7, 3),
    'dil_conv_3x3': lambda C: DilConv(C, C, 3, 2, 2),
    'dil_conv_5x5': lambda C: DilConv(C, C, 5, 4, 2),
    'dil_sep_conv_3x3': lambda C: DilSepConv(C, C, 3, 2, 2),
    'conv_1x1': lambda C: StdConv(C, C, 1, 0),
    'conv_3x1_1x3': lambda C: FacConv(C, C, 3, 1),
    'conv_3x3': lambda C: StdConv(C, C, 3, 1),
    'conv_7x1_1x7': lambda C: FacConv(C, C, 7, 3),
    'none': lambda C: Zero(),
}


class Zero(nn.Module):
    def forward(self, x, out_channels, stride):
        in_channels = x.size(1)
        if in_channels == out_channels:
            if stride == 1:
                return x.mul(0.)
            else:
                return x[:, :, ::stride, ::stride].mul(0.)
        else:
            shape = list(x.size())
            shape[1] = out_channels
            zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
            return zeros


class StdConv(nn.Module):
    def __init__(self, max_in_channels, max_out_channels, kernel_size, padding, affine=True):
        super(StdConv, self).__init__()
        self.drop_path = DropPath_()
        self.relu = nn.ReLU()
        self.conv = DynamicConv2d(max_in_channels, max_out_channels, kernel_size, 1, padding, bias=False)
        self.bn = DynamicBatchNorm2d(max_out_channels, affine=affine)

    def forward(self, x, out_channels, stride):
        x = self.drop_path(x)
        x = self.relu(x)
        x = self.conv(x, out_channels=out_channels, stride=stride)
        x = self.bn(x)
        return x


class FactorizedReduce(nn.Module):
    def __init__(self, max_in_channels, max_out_channels):
        super(FactorizedReduce, self).__init__()
        assert max_out_channels % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = DynamicConv2d(max_in_channels, max_out_channels // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = DynamicConv2d(max_in_channels, max_out_channels // 2, 1, stride=2, padding=0, bias=False)
        self.bn = DynamicBatchNorm2d(max_out_channels)

    def forward(self, x, out_channels):
        x = self.relu(x)
        assert out_channels % 2 == 0
        out = torch.cat([self.conv_1(x, out_channels=out_channels // 2),
                         self.conv_2(x[:, :, 1:, 1:], out_channels=out_channels // 2)], dim=1)
        out = self.bn(out)
        return out


class Pool(nn.Module):
    def __init__(self, pool_type, channels, kernel_size, padding, affine=True):
        super(Pool, self).__init__()
        self.pool_type = pool_type.lower()
        self.kernel_size = kernel_size
        self.padding = padding
        self.channels = channels
        # self.bn = nn.BatchNorm2d(channels, affine=affine)
        self.drop_path = DropPath_()

    def forward(self, x, out_channels, stride):
        assert out_channels <= self.channels
        if self.pool_type == 'max':
            out = F.max_pool2d(x, self.kernel_size, stride, self.padding)
        elif self.pool_type == 'avg':
            out = F.avg_pool2d(x, self.kernel_size, stride, self.padding, count_include_pad=False)
        else:
            raise ValueError
        # out = self.bn(out)
        return self.drop_path(out)


class DilConv(nn.Module):
    def __init__(self, max_in_channels, max_out_channels, kernel_size, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.relu = nn.ReLU()
        self.dw = DynamicConv2d(max_in_channels, max_in_channels, kernel_size, 1, padding,
                                dilation=dilation, groups=max_in_channels, bias=False)
        self.pw = DynamicConv2d(max_in_channels, max_out_channels, 1, stride=1, padding=0, bias=False)
        self.bn = DynamicBatchNorm2d(max_out_channels, affine=affine)
        self.drop_path = DropPath_()

    def forward(self, x, out_channels, stride):
        x = self.relu(x)
        x = self.dw(x, out_channels=x.size(1), stride=stride)
        x = self.pw(x, out_channels=out_channels)
        x = self.bn(x)
        return self.drop_path(x)


class FacConv(nn.Module):
    """
    Factorized conv
    ReLU - Conv(Kx1) - Conv(1xK) - BN
    """

    def __init__(self, max_in_channels, max_out_channels, kernel_length, padding, affine=True):
        super(FacConv, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = DynamicConv2d(max_in_channels, max_in_channels, (1, kernel_length), 1, (0, padding), bias=False)
        self.conv2 = DynamicConv2d(max_in_channels, max_out_channels, (kernel_length, 1), 1, (padding, 0), bias=False)
        self.bn = DynamicBatchNorm2d(max_out_channels, affine=affine)
        self.drop_path = DropPath_()

    def forward(self, x, out_channels, stride):
        x = self.relu(x)
        x = self.conv1(x, out_channels=x.size(1), stride=(1, stride))
        x = self.conv2(x, out_channels=out_channels, stride=(stride, 1))
        x = self.bn(x)
        return self.drop_path(x)


class StackedSepConv(nn.Module):
    """
    Separable convolution stacked twice.
    """

    def __init__(self, max_in_channels, max_out_channels, kernel_size, padding, affine=True):
        super(StackedSepConv, self).__init__()
        self.relu1 = nn.ReLU(inplace=False)
        self.dw1 = DynamicConv2d(max_in_channels, max_in_channels, kernel_size=kernel_size,
                                 stride=1, padding=padding, groups=max_in_channels, bias=False)
        self.pw1 = DynamicConv2d(max_in_channels, max_in_channels, kernel_size=1, padding=0, bias=False)
        self.bn1 = DynamicBatchNorm2d(max_in_channels, affine=affine)
        self.relu2 = nn.ReLU(inplace=False)
        self.dw2 = DynamicConv2d(max_in_channels, max_in_channels, kernel_size=kernel_size,
                                 stride=1, padding=padding, groups=max_in_channels, bias=False)
        self.pw2 = DynamicConv2d(max_in_channels, max_out_channels, kernel_size=1, padding=0, bias=False)
        self.bn2 = DynamicBatchNorm2d(max_out_channels, affine=affine)
        self.drop_path = DropPath_()

    def forward(self, x, out_channels, stride):
        x = self.relu1(x)
        x = self.dw1(x, out_channels=x.size(1), stride=stride)
        x = self.pw1(x, out_channels=x.size(1))
        x = self.bn1(x)
        x = self.relu2(x)
        x = self.dw2(x, out_channels=x.size(1))
        x = self.pw2(x, out_channels=out_channels)
        x = self.bn2(x)
        return self.drop_path(x)


class DilSepConv(nn.Module):

    def __init__(self, max_in_channels, max_out_channels, kernel_size, padding, dilation, affine=True):
        super(DilSepConv, self).__init__()
        C_in = max_in_channels
        C_out = max_out_channels
        self.relu1 = nn.ReLU(inplace=False)
        self.conv1 = DynamicConv2d(
            C_in, C_in, kernel_size=kernel_size,
            padding=padding, dilation=dilation, groups=C_in, bias=False
        )
        self.conv2 = DynamicConv2d(C_in, C_in, kernel_size=1, padding=0, bias=False)
        self.bn1 = DynamicBatchNorm2d(C_in, affine=affine)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = DynamicConv2d(
            C_in, C_in, kernel_size=kernel_size,
            padding=padding, dilation=dilation, groups=C_in, bias=False
        )
        self.conv4 = DynamicConv2d(C_in, C_out, kernel_size=1, padding=0, bias=False)
        self.bn2 = DynamicBatchNorm2d(C_out, affine=affine)
        self.drop_path = DropPath_()

    def forward(self, x, out_channels, stride):
        in_channels = x.size(1)
        x = self.relu1(x)
        x = self.conv1(x, stride=stride, out_channels=in_channels)
        x = self.conv2(x, out_channels=in_channels)
        x = self.bn1(x)
        x = self.relu2(x)
        x = self.conv3(x, out_channels=in_channels)
        x = self.conv4(x, out_channels=out_channels)
        x = self.bn2(x)
        return self.drop_path(x)


class SkipConnection(FactorizedReduce):
    def __init__(self, max_in_channels, max_out_channels):
        super().__init__(max_in_channels, max_out_channels)
        self.drop_path = DropPath_()

    def forward(self, x, out_channels, stride):
        if stride > 1:
            out = super(SkipConnection, self).forward(x, out_channels=out_channels)
            return self.drop_path(out)
        return x


#### utility layers ####

class DropPath_(nn.Module):
    # https://github.com/khanrc/pt.darts/blob/0.1/models/ops.py
    def __init__(self, drop_prob=0.):
        super(DropPath_, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.training and self.drop_prob > 0.:
            keep_prob = 1. - self.drop_prob
            mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
            return x.div(keep_prob).mul(mask)
        return x


class ReLUConvBN(nn.Module):
    def __init__(self, max_in_channels, max_out_channels, kernel_size, padding):
        super(ReLUConvBN, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv = DynamicConv2d(max_in_channels, max_out_channels, kernel_size, padding=padding, bias=False)
        self.bn = DynamicBatchNorm2d(max_out_channels)

    def forward(self, x, out_channels):
        x = self.relu(x)
        x = self.conv(x, out_channels=out_channels)
        x = self.bn(x)
        return x


import collections
import logging

import torch
import torch.nn as nn

from common.dynamic_ops import DynamicBatchNorm2d, DynamicConv2d, DynamicLinear, ResizableSequential
from common.searchspace import SearchSpace, MixedOp, MixedInput, HyperParameter
from configs import NdsConfig, NdsModelType
from .layers import OPS, DropPath_, FactorizedReduce, ReLUConvBN


logger = logging.getLogger(__name__)


class AuxiliaryHead(nn.Module):
    def __init__(self, C, num_classes):
        super(AuxiliaryHead, self).__init__()
        if num_classes == 1000:
            # assuming input size 14x14
            self.features = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
                DynamicConv2d(C, 128, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 768, 2, bias=False),
                nn.BatchNorm2d(768),
                nn.ReLU(inplace=True)
            )
        else:
            # assuming input size 8x8
            self.features = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
                DynamicConv2d(C, 128, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 768, 2, bias=False),
                nn.BatchNorm2d(768),
                nn.ReLU(inplace=True)
            )
        self.classifier = nn.Linear(768, num_classes)
        for module in self.modules():
            if isinstance(module, DynamicConv2d):
                module.allow_static_op = True

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class Cell(nn.Module):
    def __init__(self, n_nodes, primitives, C_prev_prev, C_prev, C, reduction, concat_all):
        super(Cell, self).__init__()
        self.n_nodes = n_nodes
        self.reduction = reduction
        self.cell_type = 'reduce' if reduction else 'normal'
        self.concat_all = concat_all
        self.primitives = primitives
        logger.info('Cell %s created: channels %d -> %d -> %d, %d nodes',
                    self.cell_type, C_prev_prev, C_prev, C, self.n_nodes)

        self.preprocess0_reduce = FactorizedReduce(C_prev_prev, C)
        self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 0)

        self.nodes = nn.ModuleDict()
        self.inputs = nn.ModuleDict()
        for i in range(self.n_nodes):
            self.nodes[f'{i}_x'] = self._build_layer_choice(f'{self.cell_type}_{i}_x', C)
            self.nodes[f'{i}_y'] = self._build_layer_choice(f'{self.cell_type}_{i}_y', C)
            self.inputs[f'{i}_x'] = self._build_input_choice(f'{self.cell_type}_{i}_x', i + 2)
            self.inputs[f'{i}_y'] = self._build_input_choice(f'{self.cell_type}_{i}_y', i + 2)

    def _build_input_choice(self, key, num_input_candidates):
        return MixedInput(key + '_input', num_input_candidates)

    def _build_layer_choice(self, key, channels):
        mapping = collections.OrderedDict()
        for name in self.primitives:
            mapping[name] = OPS[name](channels)
        return MixedOp(key + '_op', mapping)

    def forward(self, s0, s1, width):
        if s0.size(2) != s1.size(2):
            # needs to be down-sampled
            s0 = self.preprocess0_reduce(s0, width)
        else:
            s0 = self.preprocess0(s0, width)
        s1 = self.preprocess1(s1, width)
        states = [s0, s1]
        used_indices = set()
        for i in range(self.n_nodes):
            x_k, y_k = f'{i}_x', f'{i}_y'
            x_reduction = self.reduction and self.inputs[x_k].activated < 2
            y_reduction = self.reduction and self.inputs[y_k].activated < 2
            used_indices |= {self.inputs[x_k].activated, self.inputs[y_k].activated}
            t1 = self.nodes[x_k](self.inputs[x_k](states), width, 2 if x_reduction else 1)
            t2 = self.nodes[y_k](self.inputs[y_k](states), width, 2 if y_reduction else 1)
            states.append(t1 + t2)
        if self.concat_all:
            return torch.cat(states[2:], 1)
        else:
            unused_indices = [i for i in range(2, self.n_nodes + 2) if i not in used_indices]
            return torch.cat([states[i] for i in unused_indices], 1)


class NDS(SearchSpace):

    def __init__(self, config: NdsConfig):
        super(NDS, self).__init__()
        self.model_type = config.model_type
        self.num_labels = 10 if self.model_type == NdsModelType.CIFAR else 1000
        self.max_init_channels = max(config.init_channels)
        self.max_num_layers = max(config.num_layers)
        self.depth_selector = HyperParameter('depth', config.num_layers)
        self.width_selector = HyperParameter('width', config.init_channels)
        self.use_aux = config.use_aux
        C = self.max_init_channels

        if self.model_type == NdsModelType.ImageNet:
            self.stem0 = ResizableSequential(
                DynamicConv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
                DynamicBatchNorm2d(C // 2),
                nn.ReLU(inplace=True),
                DynamicConv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
                DynamicBatchNorm2d(C),
            )
            self.stem1 = ResizableSequential(
                nn.ReLU(inplace=True),
                DynamicConv2d(C, C, 3, stride=2, padding=1, bias=False),
                DynamicBatchNorm2d(C),
            )
            C_prev_prev, C_prev, C_curr = C, C, C
        elif self.model_type == NdsModelType.CIFAR:
            self.stem = ResizableSequential(
                DynamicConv2d(3, 3 * C, 3, padding=1, bias=False),
                DynamicBatchNorm2d(3 * C)
            )
            C_prev_prev, C_prev, C_curr = 3 * C, 3 * C, C

        self.stages = nn.ModuleList()
        for stage_idx in range(3):
            if stage_idx > 0:
                C_curr *= 2
            stage = nn.ModuleList()
            for i in range((self.max_num_layers + 2) // 3):
                cell = Cell(config.n_nodes, config.op_candidates, C_prev_prev, C_prev, C_curr,
                            stage_idx > 0 and i == 0, config.concat_all)
                stage.append(cell)
                C_prev_prev, C_prev = C_prev, cell.n_nodes * C_curr
            if stage_idx == 2:
                C_to_auxiliary = C_prev
            self.stages.append(stage)

        if self.use_aux:
            self.auxiliary_head = AuxiliaryHead(C_to_auxiliary, self.num_labels)
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = DynamicLinear(C_prev, self.num_labels)

    def forward(self, inputs):
        width = self.width_selector()
        depth = self.depth_selector()
        if self.model_type == NdsModelType.ImageNet:
            s0 = self.stem0(inputs, width / self.max_init_channels)
            s1 = self.stem1(s0, width / self.max_init_channels)
        else:
            s0 = s1 = self.stem(inputs, width / self.max_init_channels)

        cur_stage, cur_idx = 0, 0
        for i in range(depth):
            if i in [depth // 3, 2 * depth // 3]:
                width *= 2
                cur_stage += 1
                cur_idx = 0
            s0, s1 = s1, self.stages[cur_stage][cur_idx](s0, s1, width)
            if i == 2 * depth // 3:
                if self.training and self.use_aux:
                    logits_aux = self.auxiliary_head(s1)
            cur_idx += 1
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        if self.training and self.use_aux:
            return logits, logits_aux
        else:
            return logits

    def prune(self):
        super().prune()
        handler_collection = []

        def fn(m, _, __):
            m.module_used = True

        def add_hooks(m_):
            m_.module_used = False
            _handler = m_.register_forward_hook(fn)
            handler_collection.append(_handler)

        def dfs_and_delete(m):
            names = []
            for name, child in m.named_children():
                if child.module_used or isinstance(child, (nn.ModuleList, nn.ModuleDict)):
                    dfs_and_delete(child)
                    if isinstance(child, (DynamicConv2d, DynamicBatchNorm2d, DynamicLinear, ResizableSequential)):
                        child._static_mode = True
                    if isinstance(child, DynamicConv2d):
                        child.stride = child._dry_run_stride
                else:
                    names.append(name)
            for name in names:
                delattr(m, name)
            delattr(m, 'module_used')

        training = self.training
        self.eval()
        self.apply(add_hooks)
        with torch.no_grad():
            self(torch.zeros((1, 3, 32, 32)))
        for m in self.auxiliary_head.modules():
            m.module_used = True
        for handler in handler_collection:
            handler.remove()

        dfs_and_delete(self)
        self.train(training)

    def drop_path_prob(self, drop_prob):
        for module in self.modules():
            if isinstance(module, DropPath_):
                module.drop_prob = drop_prob
