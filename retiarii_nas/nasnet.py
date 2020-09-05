from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

BATCHSIZE = 32

class Pool(nn.Sequential):
    def __init__(self, pool_type, in_channels, out_channels, stride):
        super(Pool, self).__init__()
        if pool_type == "max":
            self.pool = nn.MaxPool2d(3, stride=stride, padding=1)
        elif pool_type == "avg":
            self.pool = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
        if in_channels != out_channels:
            self.conv = nn.Conv2d(in_channels, out_channels, 1, 1)
            self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)


class Identity(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride):
        super(Identity, self).__init__()
        if stride == 1 and in_channels == out_channels:
            self.identity = nn.Identity()
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, 1, stride)
            self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)


class SeparableConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, dw_kernel, dw_stride, dw_padding, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(in_channels, in_channels, dw_kernel,
                                          stride=dw_stride,
                                          padding=dw_padding,
                                          bias=bias,
                                          groups=in_channels)
        self.pointwise_conv2d = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=bias)


class BranchSeparables(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, stem=False, bias=False):
        super(BranchSeparables, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        mid_channels = out_channels if stem else in_channels
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, mid_channels, kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm2d(mid_channels, eps=0.001, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(mid_channels, out_channels, kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)


class StdConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(StdConv, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)


def _factorize_reduce_build(module, in_ch, out_ch):
    # Ugly. To align with existing pretrained checkpoints.
    module.path_1 = nn.Sequential()
    module.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
    module.path_1.add_module('conv', nn.Conv2d(in_ch, out_ch // 2, 1, stride=1, bias=False))
    module.path_2 = nn.Sequential()
    module.path_2.add_module('padding', nn.ZeroPad2d((0, 1, 0, 1)))  # to avoid nan issue, which I don't know why
    module.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
    module.path_2.add_module('conv', nn.Conv2d(in_ch, (out_ch + 1) // 2, 1, stride=1, bias=False))
    module.final_path_bn = nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.1, affine=True)


def _factorize_reduce_forward(module, x):
    x_relu = F.relu(x)
    x_path1 = module.path_1(x_relu)
    x_path2 = module.path_2(x_relu[:, :, 1:, 1:])
    return module.final_path_bn(torch.cat([x_path1, x_path2], 1))


class FactorizeReduce(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(FactorizeReduce, self).__init__()
        _factorize_reduce_build(self, in_ch, out_ch)

    def forward(self, x):
        return _factorize_reduce_forward(self, x)


OPS = {
    "sepconv3x3": lambda in_ch, out_ch, stride, stem: BranchSeparables(in_ch, out_ch, 3, stride, 1, stem=stem),
    "sepconv5x5": lambda in_ch, out_ch, stride, stem: BranchSeparables(in_ch, out_ch, 5, stride, 2, stem=stem),
    "sepconv7x7": lambda in_ch, out_ch, stride, stem: BranchSeparables(in_ch, out_ch, 7, stride, 3, stem=stem),
    "avgpool3x3": lambda in_ch, out_ch, stride, stem: Pool("avg", in_ch, out_ch, stride),
    "maxpool3x3": lambda in_ch, out_ch, stride, stem: Pool("max", in_ch, out_ch, stride),
    "identity": lambda in_ch, out_ch, stride, stem: Identity(in_ch, out_ch, stride),
}




class Cell(nn.Module):
    def __init__(self, genotype, n_hidden, in_ch, in_ch_prev, out_ch,
                 reduction=False, reduction_p=False, stem=False, channels=None):
        # Cell: cell_prev -> cell_in -> cell_out
        # Filters: in_ch_prev -> in_ch -> out_ch
        # If reduction_p, cell_prev will have twice resolution as cell_in
        # If reduction, cell_out will have half resolution of cell_in
        super(Cell, self).__init__()
        self.genotype = genotype
        self.in_ch = in_ch
        self.in_ch_prev = in_ch_prev
        self.out_ch = out_ch
        self.stem = stem
        self.operations = genotype["operations"]
        self.hidden_indices = genotype["hidden_indices"]
        self.concat_indices = genotype["concat_indices"]
        self.n_hidden = n_hidden
        assert len(self.operations) == 2 * n_hidden
        assert len(self.hidden_indices) == 2 * n_hidden

        self.reduction = reduction
        self.reduction_p = reduction_p
        if reduction_p:
            _factorize_reduce_build(self, in_ch_prev, out_ch)
        else:
            self.conv_prev_1x1 = StdConv(in_ch_prev, out_ch)
        self.conv_1x1 = StdConv(in_ch, out_ch)
        if channels is not None:
            self.channels = channels
        else:
            self.channels = [out_ch] * (self.n_hidden + 2)

        for i in range(self.n_hidden):
            i_left, i_right = self.hidden_indices[i * 2], self.hidden_indices[i * 2 + 1]
            self.add_module(f"comb_iter_{i}_left",
                            OPS[self.operations[i * 2]](self.channels[i_left], out_ch,
                                                        2 if reduction and i_left < 2 else 1, stem))
            #self.add_module(f"comb_iter_{i}_left",
            #                BranchSeparables(self.channels[i_left], out_ch,
            #                                 2 if reduction and i_left < 2 else 1, 1, stem))
            self.add_module(f"comb_iter_{i}_right",
                            OPS[self.operations[i * 2 + 1]](self.channels[i_right], out_ch,
                                                            2 if reduction and i_right < 2 else 1, stem))
            #self.add_module(f"comb_iter_{i}_right",
            #                BranchSeparables(self.channels[i_right], out_ch,
            #                                 2 if reduction and i_right < 2 else 1, 1, stem))
        for i in range(2):
            if reduction and i in self.concat_indices:
                self.add_module(f"reduction_{i}", FactorizeReduce(self.channels[i], out_ch))

    def cell(self, x, x_prev):
        # All the states (including x and x_prev) should have same filters.
        # In case of reduction, other states will have half resolution of the first two.
        # self.channels marks the filters each of the states owns.
        states = [x, x_prev]
        for i in range(self.n_hidden):
            # TODO: add input choice here
            x_left = getattr(self, f"comb_iter_{i}_left")(states[self.hidden_indices[i * 2]])
            # TODO: add input choice here
            x_right = getattr(self, f"comb_iter_{i}_right")(states[self.hidden_indices[i * 2 + 1]])
            states.append(x_left + x_right)
        for i in range(2):
            # TODO: add input or layer choice here
            if self.reduction and i in self.concat_indices:
                states[i] = getattr(self, f"reduction_{i}")(states[i])
        return torch.cat([states[i] for i in self.concat_indices], 1)

    def forward(self, x, x_prev):
        if self.reduction_p:
            x_prev = _factorize_reduce_forward(self, x_prev)
        else:
            x_prev = self.conv_prev_1x1(x_prev)
        x = self.conv_1x1(x)
        return self.cell(x, x_prev)


class CellStem0(Cell):
    def __init__(self, genotype, n_hidden, in_ch, out_ch):
        super(CellStem0, self).__init__(genotype, n_hidden, in_ch, in_ch, out_ch,
                                        reduction=True, stem=True,
                                        channels=[out_ch, in_ch] + [out_ch] * n_hidden)
        del self.conv_prev_1x1  # not used, delete to align

    def forward(self, x):
        x1 = self.conv_1x1(x)
        return self.cell(x1, x)


class CellStem1(Cell):
    def __init__(self, genotype, n_hidden, in_ch, in_ch_prev, out_ch):
        super(CellStem1, self).__init__(genotype, n_hidden, in_ch, in_ch_prev, out_ch, reduction=True, reduction_p=True)


class NASNet(nn.Module):
    def __init__(self, genotype_normal, genotype_reduce,
                 num_stem_features, num_normal_cells, filters, scaling, skip_reduction,
                 num_blocks=5, use_aux=False, num_classes=1000):
        super(NASNet, self).__init__()
        self.num_normal_cells = num_normal_cells
        self.skip_reduction = skip_reduction
        self.use_aux = use_aux
        self.num_classes = num_classes
        self.expand_normal = len(genotype_normal["concat_indices"])
        self.expand_reduce = len(genotype_reduce["concat_indices"])

        self.conv0 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(3, num_stem_features, kernel_size=3, stride=2, bias=False)),
            ('bn', nn.BatchNorm2d(num_stem_features, eps=0.001, momentum=0.1, affine=True))
        ]))

        filters_p = int(filters * scaling ** (-1))
        filters_pp = int(filters * scaling ** (-2))
        self.cell_stem_0 = CellStem0(genotype_reduce, num_blocks, num_stem_features, filters_pp)
        filters_pp *= self.expand_reduce
        self.cell_stem_1 = CellStem1(genotype_reduce, num_blocks, filters_pp, num_stem_features, filters_p)
        filters_p *= self.expand_reduce

        cell_id = 0
        for i in range(3):
            self.add_module(f'cell_{cell_id}', Cell(genotype_normal, num_blocks, filters_p, filters_pp, filters, reduction_p=True))
            cell_id += 1
            filters_pp, filters_p = filters_p, self.expand_normal * filters
            for _ in range(num_normal_cells - 1):
                self.add_module(f'cell_{cell_id}', Cell(genotype_normal, num_blocks, filters_p, filters_pp, filters))
                filters_pp = filters_p
                cell_id += 1
            if i == 1 and self.use_aux:
                self.aux_features = nn.Sequential(
                    nn.ReLU(),
                    nn.AvgPool2d(kernel_size=(5, 5), stride=(3, 3),
                                 padding=(2, 2), count_include_pad=False),
                    nn.Conv2d(in_channels=filters_p, out_channels=128, kernel_size=1, bias=False),
                    nn.BatchNorm2d(num_features=128, eps=0.001, momentum=0.1, affine=True),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=128, out_channels=768,
                              kernel_size=((14 + 2) // 3, (14 + 2) // 3), bias=False),
                    nn.BatchNorm2d(num_features=768, eps=1e-3, momentum=0.1, affine=True),
                    nn.ReLU()
                )
                self.aux_linear = nn.Linear(768, num_classes)

            filters *= scaling
            if i < 2:
                self.add_module(f'reduction_cell_{i}',
                                Cell(genotype_reduce, num_blocks, filters_p, filters_pp, filters, reduction=True))
                filters_p = self.expand_reduce * filters

        self.linear = nn.Linear(filters_p, self.num_classes)  # large: 4032; mobile: 1056

    def features(self, x):
        x_conv0 = self.conv0(x)
        x_stem_0 = self.cell_stem_0(x_conv0)
        x_stem_1 = self.cell_stem_1(x_stem_0, x_conv0)
        prev_x, x = x_stem_0, x_stem_1
        cell_id = 0
        for i in range(3):
            for _ in range(self.num_normal_cells):
                new_x = getattr(self, f'cell_{cell_id}')(x, prev_x)
                prev_x, x = x, new_x
                cell_id += 1
            if i == 1 and self.training and self.use_aux:
                x_aux = self.aux_features(x)
            if i < 2:
                new_x = getattr(self, f'reduction_cell_{i}')(x, prev_x)
                prev_x = x if not self.skip_reduction else prev_x
                x = new_x
        if self.training and self.use_aux:
            return (x, x_aux)
        return (x, None)

    def logits(self, features):
        x = F.relu(features, inplace=False)
        #print('zql: ', x.size(2))
        #x = F.avg_pool2d(x, kernel_size=int(x.size(2))).view(BATCHSIZE, -1) #x.size(0)
        x = F.avg_pool2d(x, kernel_size=int(x.size(2))).view(x.size(0), -1)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        output = self.logits(x[0])
        if self.training and self.use_aux:
            x_aux = x[1].view(x[1].size(0), -1)
            aux_output = self.aux_linear(x_aux)
            return (output, aux_output)
        return output


# _NASNET_A_NORMAL = {
#     "operations": ['sepconv5x5', 'sepconv3x3', 'sepconv5x5', 'sepconv3x3', 'avgpool3x3',
#                    'identity', 'avgpool3x3', 'avgpool3x3', 'sepconv3x3', 'identity'],
#     "hidden_indices": [0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
#     "concat_indices": [1, 2, 3, 4, 5, 6],
# }

_NASNET_A_NORMAL = {
    "operations": ['identity', 'identity', 'identity', 'identity', 'identity',
                   'identity', 'identity', 'identity', 'identity', 'identity'],
    "hidden_indices": [0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
    "concat_indices": [1, 2, 3, 4, 5, 6],
}


# _NASNET_A_REDUCE = {
#     "operations": ['sepconv5x5', 'sepconv7x7', 'maxpool3x3', 'sepconv7x7', 'avgpool3x3',
#                    'sepconv5x5', 'identity', 'avgpool3x3', 'sepconv3x3', 'maxpool3x3'],
#     "hidden_indices": [0, 1, 0, 1, 0, 1, 3, 2, 2, 0],
#     "concat_indices": [3, 4, 5, 6],
# }
_NASNET_A_REDUCE = {
    "operations": ['identity', 'identity', 'identity', 'identity', 'identity',
                   'identity', 'identity', 'identity', 'identity', 'identity'],
    "hidden_indices": [0, 1, 0, 1, 0, 1, 3, 2, 2, 0],
    "concat_indices": [3, 4, 5, 6],
}

def nasnet_a_any(operations_normal, hidden_indices_normal, concat_indices_normal,
                 operations_reduce, hidden_indices_reduce, concat_indices_reduce,
                 num_stem_features=32, num_normal_cells=4, filters=44, scaling=2, skip_reduction=False):
    return NASNet(
        {
            "operations": operations_normal,
            "hidden_indices": hidden_indices_normal,
            "concat_indices": concat_indices_normal
        },
        {
            "operations": operations_reduce,
            "hidden_indices": hidden_indices_reduce,
            "concat_indices": concat_indices_reduce
        },
        num_stem_features, num_normal_cells, filters, scaling, skip_reduction
    )

def test_random():
    import random
    random.seed(0)
    for p in range(200):
        print(p)
        genotype = {}
        for i in range(2):
            genotype[f"op{i}"] = [random.choice(list(OPS.keys())) for _ in range(10)]
            genotype[f"input{i}"] = [random.randint(0, k // 2 + 1) for k in range(10)]
            while True:
                # According to the paper, we should concatenate all unused hidden nodes.
                # But appearantly, the provided NASNet-A architecture is violating such constraints.
                # So we simply randomly choose some nodes to concat here.
                concat = [random.choice([False, True]) for _ in range(7)]
                if sum(concat) > 0:
                    concat = [i for i, k in enumerate(concat) if k]
                    break
            genotype[f"concat{i}"] = concat
        model = nasnet_a_any(genotype["op0"], genotype["input0"], genotype["concat0"],
                             genotype["op1"], genotype["input1"], genotype["concat1"])
        model(torch.randn(2, 3, 224, 224))

#====================Training approach

import sdk
from sdk.mutators.builtin_mutators import ModuleMutator
import datasets

class ModelTrain(sdk.Trainer):
    def __init__(self, device='cuda'):
        super(ModelTrain, self).__init__()
        self.device = torch.device(device)
        self.data_provider = datasets.ImagenetDataProvider(save_path="/data/v-yugzh/imagenet",
                                                    train_batch_size=BATCHSIZE,
                                                    test_batch_size=BATCHSIZE,
                                                    valid_size=None,
                                                    n_worker=4,
                                                    resize_scale=0.08,
                                                    distort_color='normal')

    def train_dataloader(self):
        return self.data_provider.train

    def val_dataloader(self):
        return self.data_provider.valid

#====================Experiment config

import random
genotype = {}
for i in range(2):
    genotype[f"op{i}"] = [random.choice(list(OPS.keys())) for _ in range(10)]
    genotype[f"input{i}"] = [random.randint(0, k // 2 + 1) for k in range(10)]
    while True:
        # According to the paper, we should concatenate all unused hidden nodes.
        # But appearantly, the provided NASNet-A architecture is violating such constraints.
        # So we simply randomly choose some nodes to concat here.
        concat = [random.choice([False, True]) for _ in range(7)]
        if sum(concat) > 0:
            concat = [i for i, k in enumerate(concat) if k]
            break
    genotype[f"concat{i}"] = concat
base_model = nasnet_a_any(genotype["op0"], genotype["input0"], genotype["concat0"],
                        genotype["op1"], genotype["input1"], genotype["concat1"])
exp = sdk.create_experiment('nasnet_search', base_model)
exp.specify_training(ModelTrain)

mutators = []
exp.specify_mutators(mutators)
exp.specify_strategy('naive.strategy.main', 'naive.strategy.RandomSampler')
run_config = {
    'authorName': 'nas',
    'experimentName': 'nas',
    'trialConcurrency': 1,
    'maxExecDuration': '24h',
    'maxTrialNum': 999,
    'trainingServicePlatform': 'local',
    'searchSpacePath': 'empty.json',
    'useAnnotation': False
} # nni experiment config
exp.run(run_config)
