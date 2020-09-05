'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import argparse
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

#from mobilenet_writer import write_metrics

trainloader = None
testloader = None
criterion = None
optimizer = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0.0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

batch_size = None
num_workers = None
num_batch = None


def prepare(net, args):
    global trainloader
    global testloader
    global criterion
    global optimizer

    # Data
    '''print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)'''

    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=args['lr'], momentum=0.9, weight_decay=5e-4)

    if args['optimizer'] == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args['lr'], momentum=0.9, weight_decay=5e-4)
    if args['optimizer'] == 'Adadelta':
        optimizer = optim.Adadelta(net.parameters(), lr=args['lr'])
    if args['optimizer'] == 'Adagrad':
        optimizer = optim.Adagrad(net.parameters(), lr=args['lr'])
    if args['optimizer'] == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=args['lr'])
    if args['optimizer'] == 'Adamax':
        optimizer = optim.Adam(net.parameters(), lr=args['lr'])


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        #self.parallel_convs = nn.Conv2d(in_planes*num_batch, in_planes*num_batch, kernel_size=1,
        #                                stride=stride, padding=0, groups=num_batch, bias=False)
        self.parallel_convs = nn.Conv2d(in_planes, in_planes, kernel_size=1,
                                        stride=stride, padding=0, groups=1, bias=False)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        bs, _, s1, s2 = x.size()
        '''if bs == batch_size:
            ori_out = self.conv1(x)
            _, _, os1, os2 = ori_out.size()
            xs = torch.cat([x] * num_batch)
            ada_out = self.parallel_convs(xs.view(batch_size, -1, s1, s2))
            out = torch.cat([ori_out] * num_batch) + ada_out.view(num_batch*batch_size, -1, os1, os2)
        else:  # 128
            ori_out = self.conv1(x)
            _, _, os1, os2 = ori_out.size()
            # print(x.size())
            ada_out = self.parallel_convs(x.view(batch_size, -1, s1, s2))
            ada_out2 = ada_out.view(num_batch*batch_size, -1, os1, os2)
            out = ori_out + ada_out2'''
        ori_out = self.conv1(x)
        ada_out = self.parallel_convs(x)
        out = ori_out + ada_out
        #----
        out = F.relu(out)
        out = F.relu(self.conv2(out))
        return out


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]
    #cfg = [64,] #(128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]

    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        #self.linear = nn.Linear(16384, num_classes)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    net = MobileNet()
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    print(y.size())

# test()


def train(net, epoch, batches=-1):
    global trainloader
    global testloader
    global criterion
    global optimizer
    global batch

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    # for batch_idx, (inputs, targets) in enumerate(trainloader):
    for batch_idx in range(100):
        inputs = torch.FloatTensor(batch_size, 3, 32, 32).random_(0, 255)
        targets = torch.LongTensor(batch_size).random_(0, 10)
        #targets = torch.cat([targets]*num_batch)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        #print('size: ', outputs.size())
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        acc = 100.*correct/total

    print(batch_idx, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

'''if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=64, help="batch size")
    parser.add_argument("--nw", type=int, default=4, help="num of workers")
    parser.add_argument("--nb", type=int, default=2, help="num trial to batch")
    parser.add_argument("--ne", type=int, default=5, help="num of epochs")
    args = parser.parse_args()
    batch_size = args.bs
    num_workers = args.nw
    num_batch = args.nb
    num_epochs = args.ne

    net = MobileNet()
    net.to(device)
    RCV_CONFIG = {'lr': 0.1, 'optimizer': 'Adam', 'model': 'mobilenet'}
    prepare(net, RCV_CONFIG)
    start_time = time.time()
    for i in range(num_epochs):
        train(net, i)
    print('batch elapsed time: ', time.time() - start_time)
    #write_metrics(args, __file__, time.time() - start_time)'''

#====================Training approach

import sdk
from sdk.mutators.builtin_mutators import ModuleMutator
import datasets

class ModelTrain(sdk.Trainer):
    def __init__(self, device='cuda'):
        super(ModelTrain, self).__init__()
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

#====================Experiment config

base_model = MobileNet()
exp = sdk.create_experiment('adaptor_search', base_model)
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
pre_run_config = {
    'name' : f'adapter',
    'x_shape' : [8, 3 , 32, 32],
    'x_dtype' : 'torch.float32',
    'y_shape' : [8],
    "y_dtype" : "torch.int64",
    "mask" : False,
    "imports" : []
}
exp.run(run_config, pre_run_config=pre_run_config)
