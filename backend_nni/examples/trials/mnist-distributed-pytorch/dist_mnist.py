# Copyright 2018 The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# NNI (https://github.com/Microsoft/nni) modified this code to show how to
# integrate distributed pytorch training with NNI SDK
#

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import nni
import logging

from math import ceil
from random import Random
from torch.autograd import Variable
from torchvision import datasets, transforms

logger = logging.getLogger('nni_pytorch_dist')

class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


class Net(nn.Module):
    """ Network architecture. """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def partition_dataset():
    """ Partitioning MNIST """
    dataset = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]))
    size = dist.get_world_size()
    bsz = 128 / float(size)
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(
        partition, batch_size=int(bsz), shuffle=True)
    return train_set, bsz


def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size


def run(params):
    """ Distributed Synchronous SGD Example """
    rank = dist.get_rank()
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    model = Net()
    model = model
    optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=params['momentum'])

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    total_loss = 0.0
    for epoch in range(3):
        epoch_loss = 0.0
        for data, target in train_set:
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()
        #logger.debug('Rank: ', rank, ', epoch: ', epoch, ': ', epoch_loss / num_batches)
        if rank == 0:
            nni.report_intermediate_result(epoch_loss / num_batches)
        total_loss += (epoch_loss / num_batches)
    total_loss /= 3
    logger.debug('Final loss: {}'.format(total_loss))
    if rank == 0:
        nni.report_final_result(total_loss)


def init_processes(fn, params, backend='tcp'):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend)
    fn(params)

def generate_default_params():
    '''
    Generate default parameters for mnist network.
    '''
    params = {
        'learning_rate': 0.01,
        'momentum': 0.5}
    return params

if __name__ == "__main__":
    RCV_PARAMS = nni.get_next_parameter()
    logger.debug(RCV_PARAMS)
    params = generate_default_params()
    params.update(RCV_PARAMS)
    init_processes(run, params)


