import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models.utils import load_state_dict_from_url

# Paper suggests 0.9997 momentum, for TensorFlow. Equivalent PyTorch momentum is
# 1.0 - tensorflow.
_BN_MOMENTUM = 1 - 0.9997
_FIRST_DEPTH = 32
_DEFAULT_DEPTHS = [16, 24, 40, 80, 96, 192, 320]
_DEFAULT_CONVOPS = ["dconv", "mconv", "mconv", "mconv", "mconv", "mconv", "mconv"]
_DEFAULT_SKIPS = [False, True, True, True, True, True, True]
_DEFAULT_KERNEL_SIZES = [3, 3, 5, 5, 3, 5, 3]
_DEFAULT_NUM_LAYERS = [1, 3, 3, 3, 2, 4, 1]


class _ResidualBlock(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x) + x


class _InvertedResidual(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride, expansion_factor, skip, bn_momentum=0.1):
        super(_InvertedResidual, self).__init__()
        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        mid_ch = in_ch * expansion_factor
        self.apply_residual = skip and in_ch == out_ch and stride == 1
        self.layers = nn.Sequential(
            # Pointwise
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # Depthwise
            nn.Conv2d(mid_ch, mid_ch, kernel_size, padding=kernel_size // 2,
                      stride=stride, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # Linear pointwise. Note that there's no activation.
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=bn_momentum))

    def forward(self, input):
        if self.apply_residual:
            return self.layers(input) + input
        else:
            return self.layers(input)


def _stack_inverted_residual(in_ch, out_ch, kernel_size, skip, stride, exp_factor, repeats, bn_momentum):
    """ Creates a stack of inverted residuals. """
    assert repeats >= 1
    # First one has no skip, because feature map size changes.
    first = _InvertedResidual(in_ch, out_ch, kernel_size, stride, exp_factor, skip, bn_momentum=bn_momentum)
    remaining = []
    for _ in range(1, repeats):
        remaining.append(_InvertedResidual(out_ch, out_ch, kernel_size, 1, exp_factor, skip, bn_momentum=bn_momentum))
    return nn.Sequential(first, *remaining)


def _stack_normal_conv(in_ch, out_ch, kernel_size, skip, dconv, stride, repeats, bn_momentum):
    assert repeats >= 1
    stack = []
    for i in range(repeats):
        s = stride if i == 0 else 1
        if dconv:
            modules = [
                nn.Conv2d(in_ch, in_ch, kernel_size, padding=kernel_size // 2, stride=s, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch, momentum=bn_momentum),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, out_ch, 1, padding=0, stride=1, bias=False),
                nn.BatchNorm2d(out_ch, momentum=bn_momentum)
            ]
        else:
            modules = [
                nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, stride=s, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_ch, momentum=bn_momentum)
            ]
        if skip and in_ch == out_ch and s == 1:
            # use different implementation for skip and noskip to align with pytorch
            stack.append(_ResidualBlock(nn.Sequential(*modules)))
        else:
            stack += modules
        in_ch = out_ch
    return stack


def _round_to_multiple_of(val, divisor, round_up_bias=0.9):
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


def _get_depths(depths, alpha):
    """ Scales tensor depths as in reference MobileNet code, prefers rouding up
    rather than down. """
    return [_round_to_multiple_of(depth * alpha, 8) for depth in depths]

from sdk.custom_ops_torch import Placeholder

class Block(torch.nn.Module):
    def __init__(self, block_id):
        super().__init__()
        self.placeholder = Placeholder('block_'+str(block_id)+'_0')

    def forward(self, x):
        return self.placeholder(x)

class MNASNet(torch.nn.Module):
    """ MNASNet, as described in https://arxiv.org/pdf/1807.11626.pdf. This
    implements the B1 variant of the model.
    >>> model = MNASNet(1000, 1.0)
    >>> x = torch.rand(1, 3, 224, 224)
    >>> y = model(x)
    >>> y.dim()
    1
    >>> y.nelement()
    1000
    """
    # Version 2 adds depth scaling in the initial stages of the network.
    _version = 2

    def __init__(self, alpha, depths, convops, kernel_sizes, num_layers,
                 skips, num_classes=1000, dropout=0.2):
        super(MNASNet, self).__init__()
        assert alpha > 0.0
        assert len(depths) == len(convops) == len(kernel_sizes) == len(num_layers) == len(skips) == 7
        self.alpha = alpha
        self.num_classes = num_classes
        depths = _get_depths([_FIRST_DEPTH] + depths, alpha)
        exp_ratios = [3, 3, 3, 6, 6, 6, 6]
        strides = [1, 2, 2, 2, 1, 2, 1]
        layers = [
            # First layer: regular conv.
            nn.Conv2d(3, depths[0], 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(depths[0], momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
        ]
        for i in range(7):
            layers.append(Block(i))
        '''for conv, prev_depth, depth, ks, skip, stride, repeat, exp_ratio in \
                zip(convops, depths[:-1], depths[1:], kernel_sizes, skips, strides, num_layers, exp_ratios):
            if conv == "mconv":
                # MNASNet blocks: stacks of inverted residuals.
                layers.append(_stack_inverted_residual(prev_depth, depth, ks, skip,
                                                       stride, exp_ratio, repeat, _BN_MOMENTUM))
            else:
                # Normal conv and depth-separated conv
                layers += _stack_normal_conv(prev_depth, depth, ks, skip, conv == "dconv",
                                             stride, repeat, _BN_MOMENTUM)'''
        layers += [
            # Final mapping to classifier input.
            #nn.Conv2d(depths[7], 1280, 1, padding=0, stride=1, bias=False),
            nn.Conv2d(depths[0], 1280, 1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(1280, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
        ]
        self.layers = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Dropout(p=dropout, inplace=True),
                                        nn.Linear(1280, num_classes))
        self._initialize_weights()

    def forward(self, x):
        x = self.layers(x)
        # Equivalent to global avgpool and removing H and W dimensions.
        x = x.mean([2, 3])
        return self.classifier(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_out",
                                         nonlinearity="sigmoid")
                nn.init.zeros_(m.bias)


def test_model(model):
    model(torch.randn(2, 3, 224, 224))

#====================definition of candidate op classes
BN_MOMENTUM = 1 - 0.9997

class RegularConv(nn.Module):
    def __init__(self, kernel_size, in_ch, out_ch, skip, exp_ratio, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.skip = skip
        self.exp_ratio = exp_ratio
        self.stride = stride

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, stride=stride, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_ch, momentum=BN_MOMENTUM)

    def forward(self, x):
        out = self.bn(self.relu(self.conv(x)))
        if self.skip == 'identity':
            out = out + x
        return out

class DepthwiseConv(nn.Module):
    def __init__(self, kernel_size, in_ch, out_ch, skip, exp_ratio, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.skip = skip
        self.exp_ratio = exp_ratio
        self.stride = stride

        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size, padding=kernel_size // 2, stride=stride, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 1, padding=0, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch, momentum=BN_MOMENTUM)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.skip == 'identity':
            out = out + x
        return out

class MobileConv(nn.Module):
    def __init__(self, kernel_size, in_ch, out_ch, skip, exp_ratio, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.skip = skip
        self.exp_ratio = exp_ratio
        self.stride = stride

        mid_ch = in_ch * exp_ratio
        self.layers = nn.Sequential(
            # Pointwise
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            # Depthwise
            nn.Conv2d(mid_ch, mid_ch, kernel_size, padding=kernel_size // 2,
                      stride=stride, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            # Linear pointwise. Note that there's no activation.
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=BN_MOMENTUM))

    def forward(self, x):
        out = self.layers(x)
        if self.skip == 'identity':
            out = out + x
        return out

from sdk.mutators.mutator import Mutator
from sdk.graph import Node, NodeType
class BlockMutator(Mutator):
    def __init__(self, seq, target: str,
                 n_layer_options: 'List',
                 op_type_options: 'List',
                 kernel_size_options: 'List',
                 se_ratio_options: 'List',
                 skip_options: 'List',
                 n_filter_options: 'List',
                 exp_ratio, stride):
        self.seq = seq
        self.target = target
        self.n_layer_options = n_layer_options
        self.op_type_options = op_type_options
        self.kernel_size_options = kernel_size_options
        self.se_ratio_options = se_ratio_options # TODO: ignore se for now
        self.skip_options = skip_options
        self.n_filter_options = n_filter_options
        self.exp_ratio = exp_ratio
        self.stride = stride

    def retrieve_targeted_graph(self, graph: 'Graph') -> 'Graph':
        return graph.find_node(self.target)

    def mutate(self, graph):
        target_node = self.retrieve_targeted_graph(graph)
        predecessors = graph.get_predecessors(target_node)
        assert len(predecessors) == 1
        if 'out_ch' in predecessors[0].operation.params:
            in_channel_num = predecessors[0].operation.params['out_ch']
        else:
            in_channel_num = 16
        # generate a new layer
        op_type = self.choice(self.op_type_options)
        kernel_size = self.choice(self.kernel_size_options)
        n_filter = self.choice(self.n_filter_options)
        skip = self.choice(self.skip_options)
        # replace target_node with the new layer
        target_node.update_operation(op_type, **{
            'kernel_size': kernel_size,
            'in_ch': in_channel_num,
            'out_ch': n_filter,
            'skip': 'no', # no skip for the first layer
            'exp_ratio': self.exp_ratio,
            'stride': self.stride})
        # insert x new layers after target_node
        n_layer = self.choice(self.n_layer_options)
        for i in range(1, n_layer):
            new_node = Node(graph, NodeType.Hidden, self.target+'_'+str(i))
            new_node.update_operation(op_type, **{
                'kernel_size': kernel_size,
                'in_ch': n_filter,
                'out_ch': n_filter,
                'skip': skip,
                'exp_ratio': self.exp_ratio,
                'stride': 1})
            graph.insert_after(target_node, new_node)
            target_node = new_node
        if self.seq == 9:
            successors = graph.get_successors(target_node)
            assert len(successors) == 1
            successors[0].update_operation(None, in_channels=n_filter)
