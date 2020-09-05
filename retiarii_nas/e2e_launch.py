from collections import OrderedDict
import shutil
import subprocess
import sys
import time
import threading

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sdk.mutators import Mutator
from sdk.translate_code import gen_pytorch_graph

import collections
from transformers import BertModel, BertTokenizer

class SharedDataLoader(object):
    def __init__(self, dataset, rank, once, batch_size, num_workers, **kwargs):
        self.num_workers = num_workers
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, **kwargs)
        if rank == 0 or not once:
            self.model = BertModel.from_pretrained('bert-base-uncased').cuda()
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_queue = 32
        self.once = once
        self.input_size = (batch_size, 64, 768)
        self.batch_size = batch_size
        self.rank = rank
        self.length = len(dataset) // batch_size

    def __iter__(self):
        self.counter = 0
        self.shared_queue = collections.deque()
        if self.rank == 0 or not self.once:
            if self.num_workers == 0:
                return self._process_gen()
            else:
                threading.Thread(target=self._process).start()
        return self

    def __len__(self):
        return len(self.dataloader)

    def _data_preprocess(self, text, label):
        text = torch.tensor([self.tokenizer.encode(t, max_length=64, pad_to_max_length=True) for t in text]).cuda()
        mask = text > 0
        with torch.no_grad():
            output, _ = self.model(text)
        return output, mask, label.cuda()

    def _process_gen(self):
        for text, label in self.dataloader:
            yield self._data_preprocess(text, label)

    def _process(self):
        for text, label in self.dataloader:
            while len(self.shared_queue) >= self.max_queue:
                time.sleep(1)
            data = self._data_preprocess(text, label)            
            self.shared_queue.append(data)

    def __next__(self):
        self.counter += 1
        if self.counter >= len(self):
            raise StopIteration
        if not self.once:
            while not self.shared_queue:
                time.sleep(0.1)
            return self.shared_queue.popleft()
        if self.rank == 0:
            while not self.shared_queue:
                time.sleep(0.1)
            text, masks, labels = self.shared_queue.popleft()
            masks = masks.float()
        else:
            text = torch.zeros(self.input_size, dtype=torch.float, device="cuda")
            labels = torch.zeros(self.batch_size, dtype=torch.long, device="cuda")
            masks = torch.zeros(self.input_size[:2], dtype=torch.float, device="cuda")
        torch.distributed.broadcast(text, 0)
        torch.distributed.broadcast(labels, 0)
        torch.distributed.broadcast(masks, 0)
        masks = masks.bool()
        return text, masks, labels

#====================Training approach

import sdk
import datasets

class ModelTrain(sdk.Trainer):
    def __init__(self, device='cuda'):
        super(ModelTrain, self).__init__()
        self.n_epochs = 1
        self.device = torch.device(device)
        self.data_provider = datasets.ImagenetDataProvider(save_path="/data/v-yugzh/imagenet",
                                                    train_batch_size=8,
                                                    test_batch_size=8,
                                                    valid_size=None,
                                                    n_worker=4,
                                                    resize_scale=0.08,
                                                    distort_color='normal')

    def train_dataloader(self):
        return self.data_provider.train

    def val_dataloader(self):
        return self.data_provider.valid

    def configure_optimizer(self):
        return optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5)

    def train_step(self, x, y, infer_y):
        assert self.model is not None
        assert self.optimizer is not None
        loss = F.cross_entropy(infer_y, y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def train(self):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_dataloader()):
            data, target = data.to(self.device), target.to(self.device)
            infer_target = self.model(data)
            print('step: {}'.format(batch_idx))
            self.train_step(data, target, infer_target)
            break

    def validate(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        test_loader = self.val_dataloader()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                # sum up batch loss
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                break
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        return accuracy

import torchvision
import torchvision.transforms as transforms

class ModelTrainCifar(sdk.Trainer):
    def __init__(self, device='cuda'):
        super(ModelTrainCifar, self).__init__()
        self.n_epochs = 1
        self.device = torch.device(device)

    def train_dataloader(self):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4, drop_last=True)
        return trainloader

    def val_dataloader(self):
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
        return testloader

    def configure_optimizer(self):
        return optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5)

    def train_step(self, x, y, infer_y):
        assert self.model is not None
        assert self.optimizer is not None
        loss = F.cross_entropy(infer_y, y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def train(self):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_dataloader()):
            data, target = data.to(self.device), target.to(self.device)
            infer_target = self.model(data)
            print('step: {}'.format(batch_idx))
            self.train_step(data, target, infer_target)
            break

    def validate(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        test_loader = self.val_dataloader()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                # sum up batch loss
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                break
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        return accuracy

import textnas_dataset
class ModelTrainTextNAS(sdk.Trainer):
    def __init__(self, device='cuda'):
        super(ModelTrainTextNAS, self).__init__()
        self.n_epochs = 1
        self.device = torch.device(device)
        train_dataset, _, test_dataset = textnas_dataset.read_data_sst(train_with_valid=True)
        self.train_loader = SharedDataLoader(train_dataset, -1, False, batch_size=256, num_workers=0, shuffle=True, drop_last=True)
        self.test_loader = SharedDataLoader(test_dataset, -1, False, batch_size=256, num_workers=0)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.test_loader

    def configure_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=5E-4, eps=1E-3, weight_decay=3E-6)

    def train(self):
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for step, (text, mask, label) in enumerate(self.train_dataloader()):
            bs = text.size(0)
            self.optimizer.zero_grad()
            logits = self.model(text, mask)
            loss = criterion(logits, label)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
            self.optimizer.step()
            #acc = accuracy(logits, label)
            break

    def validate(self):
        return 0.0

#====================Experiment config

def main(nas_mode, strategy_name='random'):
    if strategy_name == 'random':
        strategy_config = ['naive.strategy.main', 'naive.strategy.RandomSampler']
    elif strategy_name == 'rl':
        strategy_config = ['strategies.rl.strategy.main', 'strategies.rl.strategy.DeterministicSampler']
    elif strategy_name == 'evolution':
        strategy_config = ['strategies.evo.strategy.main', 'strategies.evo.strategy.MutatorSampler']
    else:
        raise RuntimeError('Unrecognized strategy.')

    if nas_mode == 'nasnet_e2e':
        from examples.allstars.searchspace.nasnet import nasnet_a_mobile, OPS, Cell, CellMutator
        base_model = nasnet_a_mobile()
        exp = sdk.create_experiment('nasnet_search', base_model)
        cells = []
        for name, module in base_model.named_modules():
            if isinstance(module, Cell):
                cells.append(name)
        collapsed_nodes = {name: 'Cell' for name in cells}
        exp.specify_collapsed_nodes(collapsed_nodes)
        mutators = [CellMutator(cells, num_nodes=5, operation_candidates=list(OPS.keys()))]
        exp.specify_training(ModelTrain)
    elif nas_mode == 'mnasnet_e2e':
        _DEFAULT_DEPTHS = [16, 24, 40, 80, 96, 192, 320]
        _DEFAULT_CONVOPS = ["dconv", "mconv", "mconv", "mconv", "mconv", "mconv", "mconv"]
        _DEFAULT_SKIPS = [False, True, True, True, True, True, True]
        _DEFAULT_KERNEL_SIZES = [3, 3, 5, 5, 3, 5, 3]
        _DEFAULT_NUM_LAYERS = [1, 3, 3, 3, 2, 4, 1]
        from examples.allstars.searchspace.mnasnet import MNASNet, BlockMutator
        base_model = MNASNet(0.5, _DEFAULT_DEPTHS, _DEFAULT_CONVOPS, _DEFAULT_KERNEL_SIZES,
                            _DEFAULT_NUM_LAYERS, _DEFAULT_SKIPS)
        exp = sdk.create_experiment('mnasnet_search', base_model)
        mutators = []
        base_filter_sizes = [16, 24, 40, 80, 96, 192, 320]
        exp_ratios = [3, 3, 3, 6, 6, 6, 6]
        strides = [1, 2, 2, 2, 1, 2, 1]
        for i in range(3, 10):
            mutators.append(BlockMutator(i, 'layers__'+str(i)+'__placeholder',
                            n_layer_options=[1, 2, 3, 4],
                            op_type_options=['RegularConv', 'DepthwiseConv', 'MobileConv'],
                            kernel_size_options=[3, 5],
                            se_ratio_options=[0, 0.25],
                            #skip_options=['pool', 'identity', 'no'],
                            skip_options=['identity', 'no'],
                            n_filter_options=[int(base_filter_sizes[i-3]*x) for x in [0.75, 1.0, 1.25]],
                            exp_ratio = exp_ratios[i-3],
                            stride = strides[i-3]))
        exp.specify_training(ModelTrain)
    elif nas_mode == 'amoebanet_e2e':
        from examples.allstars.searchspace.nasnet import nasnet_a_mobile, OPS, Cell, CellMutator
        base_model = nasnet_a_mobile()
        exp = sdk.create_experiment('nasnet_search', base_model)
        cells = []
        for name, module in base_model.named_modules():
            if isinstance(module, Cell):
                cells.append(name)
        collapsed_nodes = {name: 'Cell' for name in cells}
        exp.specify_collapsed_nodes(collapsed_nodes)
        mutators = [CellMutator(cells, num_nodes=5, operation_candidates=list(OPS.keys()))]
        exp.specify_training(ModelTrain)
        strategy_config = ['strategies.evo.strategy.main', 'strategies.evo.strategy.MutatorSampler']
    elif nas_mode == 'proxylessnas_e2e':
        from examples.allstars.searchspace.proxylessnas import MobileNetV2, BlockMutator, InvertedResidual
        base_model = MobileNetV2()
        exp = sdk.create_experiment('proxylessnas_search', base_model)
        blocks = []
        for name, module in base_model.named_modules():
            if isinstance(module, InvertedResidual):
                blocks.append(name)
        collapsed_nodes = {name: 'InvertedResidual' for name in blocks}
        exp.specify_collapsed_nodes(collapsed_nodes)
        exp.specify_training(ModelTrain)
        blocks = blocks[1:]  # discard the first block
        mutators = [BlockMutator(block, [3, 6], [3, 5, 7], None, 'allow') for block in blocks]
    elif nas_mode == 'chamnet_e2e':
        from examples.allstars.searchspace.proxylessnas import MobileNetV2, BlockMutator, InvertedResidual
        base_model = MobileNetV2()
        exp = sdk.create_experiment('chamnet_search', base_model)
        blocks = []
        for name, module in base_model.named_modules():
            if isinstance(module, InvertedResidual):
                blocks.append(name)
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
        collapsed_nodes = {name: 'InvertedResidual' for name in blocks}
        exp.specify_collapsed_nodes(collapsed_nodes)
        exp.specify_training(ModelTrain)
        blocks = blocks[1:]  # discard the first block
        mutators = [BlockMutator(block, [2, 3, 4, 5, 6], None, channel_ranges[i], allow_skips[i])
                    for i, block in enumerate(blocks)]
    elif nas_mode == 'onceforall_e2e':
        from examples.allstars.searchspace.proxylessnas import MobileNetV2, BlockMutator, InvertedResidual
        base_model = MobileNetV2()
        exp = sdk.create_experiment('chamnet_search', base_model)
        blocks = []
        for name, module in base_model.named_modules():
            if isinstance(module, InvertedResidual):
                blocks.append(name)
        allow_skips = [
            None,
            None, None, 'allow', 'allow',
            None, None, 'allow', 'allow',
            None, None, 'allow', 'allow',
            None, None, 'allow', 'allow',
            None, None, 'allow', 'allow',
            None
        ]
        collapsed_nodes = {name: 'InvertedResidual' for name in blocks}
        exp.specify_collapsed_nodes(collapsed_nodes)
        exp.specify_training(ModelTrain)
        mutators = [BlockMutator(block, [3, 4, 6], [3, 5, 7], None, allow_skips[i])
                    for i, block in enumerate(blocks)]
    elif nas_mode == 'singlepathnas_e2e':
        from examples.allstars.searchspace.proxylessnas import MobileNetV2, BlockMutator, InvertedResidual
        base_model = MobileNetV2()
        exp = sdk.create_experiment('singlepathnas_search', base_model)
        blocks = []
        for name, module in base_model.named_modules():
            if isinstance(module, InvertedResidual):
                blocks.append(name)
        blocks = blocks[1:]  # discard the first block
        collapsed_nodes = {name: 'InvertedResidual' for name in blocks}
        exp.specify_collapsed_nodes(collapsed_nodes)
        exp.specify_training(ModelTrain)
        mutators = [BlockMutator(block, [3, 6], [3, 5], None, 'allow') for block in blocks]
    elif nas_mode == 'nasbench101_e2e':
        from examples.allstars.searchspace.nasbench101 import Nb101Network, Cell, CellMutator
        base_model = Nb101Network()
        exp = sdk.create_experiment('nasbench101_search', base_model)
        cells = []
        for name, module in base_model.named_modules():
            if isinstance(module, Cell):
                cells.append(name)
        collapsed_nodes = {name: 'Cell101' for name in cells}
        exp.specify_collapsed_nodes(collapsed_nodes)
        mutators = [CellMutator(cells)]
        exp.specify_training(ModelTrainCifar)
    elif nas_mode == 'nasbench201_e2e':
        from examples.allstars.searchspace.nasbench201 import Nb201Network, Cell, CellMutator
        base_model = Nb201Network()
        exp = sdk.create_experiment('nasbench201_search', base_model)
        cells = []
        for name, module in base_model.named_modules():
            if isinstance(module, Cell):
                cells.append(name)
        collapsed_nodes = {name: 'Cell201' for name in cells}
        exp.specify_collapsed_nodes(collapsed_nodes)
        mutators = [CellMutator(cells)]
        exp.specify_training(ModelTrainCifar)
    elif nas_mode == 'fbnet_e2e':
        from examples.allstars.searchspace.fbnet import fbnet_base, WrapperOp, BlockMutator
        base_model = fbnet_base("fbnet_no_skip", pretrained=False)  # choose this as it skips no ops
        exp = sdk.create_experiment('fbnet_search', base_model)
        blocks = []
        for name, module in base_model.named_modules():
            if isinstance(module, WrapperOp):
                blocks.append(name)
        blocks = blocks[1:-1]  # pop the first one and the last one
        collapsed_nodes = {name: 'WrapperOp' for name in blocks}
        exp.specify_collapsed_nodes(collapsed_nodes)
        mutators = [BlockMutator(block) for block in blocks]
        exp.specify_training(ModelTrain)
    elif nas_mode == 'spos':
        # build super graph, not executable, should be executed with mixed parallelism!!!
        from examples.allstars.searchspace.spos import ShuffleNetV2OneShot
        from sdk.mutators.builtin_mutators import ModuleMutator
        base_model = ShuffleNetV2OneShot()
        exp = sdk.create_experiment('spos_search', base_model)
        mutators = []
        for i in range(20):
            mutator = ModuleMutator('features.'+str(i), [{'ksize': 3}, {'ksize': 5}, {'ksize': 7}, {'ksize': 3, 'sequence': 'dpdpdp'}])
            mutators.append(mutator)
        collapsed_nodes = 'spos'
        exp.specify_collapsed_nodes(collapsed_nodes)
        exp.specify_training(ModelTrain)
    elif nas_mode == 'textnas_e2e':
        from examples.allstars.searchspace.textnas.textnas import Model, LayerMutator
        base_model = Model()
        exp = sdk.create_experiment('textnas_search', base_model)
        mutators = []
        for i in range(24):
            candidates = [{'op_choice': 'RNN', 'input_args': [256, 0.5]},
                        {'op_choice': 'Attention', 'input_args': [256, 4, 0.5, True]},
                        {'op_choice': 'MaxPool', 'input_args': [3, False, True]},
                        {'op_choice': 'AvgPool', 'input_args': [3, False, True]},
                        {'op_choice': 'conv_shortcut7', 'input_args': [7, 256, 0.5]},
                        {'op_choice': 'conv_shortcut5', 'input_args': [5, 256, 0.5]},
                        {'op_choice': 'conv_shortcut3', 'input_args': [3, 256, 0.5]}]
            mutators.append(LayerMutator(i, 'layers.'+str(i)+'.op', candidates))
        collapsed_nodes = 'textnas'
        exp.specify_collapsed_nodes(collapsed_nodes)
        exp.specify_training(ModelTrainTextNAS)
    elif nas_mode in ['hierarchical', 'wann', 'path_level']:
        print(f'Error: {nas_mode} should launch directly from retiarii.py')
        #raise ValueError('f{nas_mode} should launch directly from retiarii.py')
    else:
        raise RuntimeError('Unrecognized NAS mode.')

    exp.specify_mutators(mutators)
    exp.specify_strategy(*strategy_config)
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


# candidates: nasnet_e2e, mnasnet_e2e, amoebanet_e2e, proxylessnas_e2e, nasbench101_e2e, nasbench201_e2e, fbnet_e2e, spos, textnas_e2e
#nas_mode = 'nasnet_e2e'
# candidates: random, rl, evolution
#strategy_name = 'random'
#main(nas_mode, strategy_name)
