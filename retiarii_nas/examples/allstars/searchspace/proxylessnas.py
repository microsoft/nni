import torch
import torch.nn as nn

from sdk.graph import Graph
from sdk.mutators import Mutator
from sdk.translate_code import gen_pytorch_graph

__all__ = ['proxylessnas', 'chamnet', 'onceforall', 'singlepathnas', 'InvertedResidual',
           'proxylessnas_gradient']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, kernel_size):
        super(InvertedResidual, self).__init__()
        self.inp = inp
        self.oup = oup
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.kernel_size = kernel_size
        norm_layer = nn.BatchNorm2d

        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 norm_layer=None):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 4, 2],
                [6, 32, 4, 2],
                [6, 64, 4, 2],
                [6, 96, 4, 1],
                [6, 160, 4, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride,
                                      kernel_size=3, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


class BlockMutator(Mutator):
    def __init__(self, target, expand_ratios, kernel_sizes, channels, skip, differentiable=False):
        self.target = target
        self.expand_ratios = expand_ratios
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        self.skip = skip
        self.differentiable = differentiable

    def _infer_output_channels(self, graph, node):
        if node.operation.type == 'InvertedResidual':
            return node.operation.params['oup']
        if node.operation.type == 'BatchNorm2d':
            return node.operation.params['num_features']
        if node.operation.type == 'ReLU6':
            return self._infer_output_channels(graph, graph.get_predecessors(node)[0])
        raise AssertionError('Un+supported operation %s' % node.operation)

    def _update_successor_in_dim(self, graph, target_node, out_channels):
        for node in graph.get_successors(target_node):
            if node.operation.type in ('aten::relu', 'aten::relu6', 'aten::avg_pool2d', 'aten::view', 'aten::dropout'):
                self._update_successor_in_dim(graph, node, out_channels)
            elif node.operation.type == 'Linear':
                node.update_operation(None, in_features=out_channels)
            elif node.operation.type == 'BatchNorm2d':
                node.update_operation(None, out_features=out_channels)
                self._update_successor_in_dim(graph, node, out_channels)
            elif node.operation.type == 'Conv2d':
                node.update_operation(None, in_channels=out_channels)

    def mutate(self, graph):
        target_node = graph.find_node(self.target)
        params = target_node.operation.params
        if self.differentiable:
            dumped_operation = target_node.operation.dump()
            target_node.update_operation('ProxylessNASMixedOp',
                operations=[
                    {
                        **dumped_operation,
                        'kernel_size': ks,
                        'expand_ratio': ex
                    }
                    for ks in self.kernel_sizes for ex in self.expand_ratios
                ]
            )
            return
        if params['stride'] == 1 and params['inp'] == params['oup']:
            is_skipped = False
            if self.skip == 'allow':
                is_skipped = self.choice([False, True])
            elif self.skip == 'must':
                is_skipped = True
            if is_skipped:
                previous_node = graph.get_predecessors(target_node)[0]
                next_node = graph.get_successors(target_node)[0]
                graph.remove_node(target_node)
                graph.add_edge(previous_node, next_node)
                return
        new_kwargs = {}
        if self.expand_ratios is not None:
            expand_ratio = self.choice(self.expand_ratios)
            new_kwargs['expand_ratio'] = expand_ratio
        if self.kernel_sizes is not None:
            kernel_size = self.choice(self.kernel_sizes)
            new_kwargs['kernel_size'] = kernel_size
        if self.channels is not None:
            channel = self.choice(self.channels)
            new_kwargs['oup'] = channel
            previous_nodes = graph.get_predecessors(target_node)
            if previous_nodes:
                new_kwargs['inp'] = self._infer_output_channels(graph, previous_nodes[0])
        target_node.update_operation(None, **new_kwargs)
        if self.channels is not None:
            self._update_successor_in_dim(graph, target_node, new_kwargs['oup'])


def _mobilenet_v2_and_blocks():
    model = MobileNetV2()

    blocks = []
    for name, module in model.named_modules():
        if isinstance(module, InvertedResidual):
            blocks.append(name)

    model_graph = gen_pytorch_graph(model, dummy_input=(torch.randn(1, 3, 224, 224),),
                                    collapsed_nodes={name: 'InvertedResidual' for name in blocks})
    return model_graph, blocks


def proxylessnas():
    model_graph, blocks = _mobilenet_v2_and_blocks()
    blocks = blocks[1:]  # discard the first block
    mutators = [BlockMutator(block, [3, 6], [3, 5, 7], None, 'allow') for block in blocks]
    return model_graph, mutators


def proxylessnas_gradient():
    model_graph, blocks = _mobilenet_v2_and_blocks()
    blocks = blocks[1:]  # discard the first block
    mutators = [BlockMutator(block, [3, 6], [3, 5, 7], None, 'allow', differentiable=True) for block in blocks]
    graph = Graph.load(model_graph)
    for mutator in mutators:
        graph = mutator.apply(graph, None)
    return graph.dump()


def chamnet():
    model_graph, blocks = _mobilenet_v2_and_blocks()
    # TODO: first conv and last conv mutator
    channel_ranges = \
        [list(range(8, 32 + 1))] + \
        [list(range(8, 40 + 1))] * 4 + \
        [list(range(8, 48 + 1))] * 4 + \
        [list(range(16, 96 + 1))] * 4 + \
        [list(range(32, 160 + 1))] * 4 + \
        [list(range(56, 256 + 1))] * 4 + \
        [list(range(96, 480 + 1))]
    allow_skips = [
        None,
        None, 'allow', 'must', 'must',
        None, 'allow', 'allow', 'must',
        None, 'allow', 'allow', 'allow',
        None, 'allow', 'allow', 'must',
        None, 'allow', 'allow', 'must',
        None
    ]

    mutators = [BlockMutator(block, [2, 3, 4, 5, 6], None, channel_ranges[i], allow_skips[i])
                for i, block in enumerate(blocks)]
    return model_graph, mutators


def onceforall():
    model_graph, blocks = _mobilenet_v2_and_blocks()
    allow_skips = [
        None,
        None, None, 'allow', 'allow',
        None, None, 'allow', 'allow',
        None, None, 'allow', 'allow',
        None, None, 'allow', 'allow',
        None, None, 'allow', 'allow',
        None
    ]

    mutators = [BlockMutator(block, [3, 4, 6], [3, 5, 7], None, allow_skips[i])
                for i, block in enumerate(blocks)]
    return model_graph, mutators


def singlepathnas():
    model_graph, blocks = _mobilenet_v2_and_blocks()
    blocks = blocks[1:]  # discard the first block
    mutators = [BlockMutator(block, [3, 6], [3, 5], None, 'allow') for block in blocks]
    return model_graph, mutators
