# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Fast and lightweight linear regression solver to test
# adaptdl functionality.
# From https://github.com/pytorch/examples/blob/master/regression/main.py


from __future__ import print_function
from itertools import count
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

import argparse
import adaptdl
import adaptdl.torch as et


from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F

import nni


IS_CHIEF = int(os.getenv("ADAPTDL_RANK", "0")) == 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Trainer(object):
    def __init__(self, model, optim, lr_scheduler):
        self.model = model
        model.train()
        self.optim = optim
        self.lr_scheduler = lr_scheduler

        self.criterion = nn.CrossEntropyLoss()

    # Training
    def train(self, inputs, targets, stats):
        inputs, targets = inputs.to(device), targets.to(device)
        self.optim.zero_grad()
        outputs = self.model(inputs)
        loss = F.smooth_l1_loss(outputs, targets)
        loss.backward()
        self.optim.step()

        stats["loss_sum"] += loss.item() * targets.size(0)
        stats["total"] += targets.size(0)
        return {"Loss": loss.item()}


def main(params):
    et.init_process_group("nccl" if torch.cuda.is_available()
                        else "gloo")

    POLY_DEGREE = 4
    W_target = torch.randn(POLY_DEGREE, 1) * 5
    b_target = torch.randn(1) * 5


    # Define model
    fc = torch.nn.Linear(W_target.size(0), 1)

    net = fc.to(device)
    if device == 'cuda':
        cudnn.benchmark = True

    if params["autoscale_bsz"]:
        max_batch_size = 8 * params["bs"]
        local_bsz_bounds = (32,1024)
    else:
        max_batch_size = None
        local_bsz_bounds = None

    def make_features(x):
        """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
        x = x.unsqueeze(1)
        return torch.cat([x ** i for i in range(1, POLY_DEGREE+1)], 1)

    def f(x):
        """Approximated function."""
        return x.mm(W_target) + b_target.item()

    class SimpleDataset(Dataset):
        def __init__(self, size):
            random = torch.randn(size)
            x = make_features(random)
            y = f(x) + 0.25 * torch.randn(1)
            self.data = list(zip(x, y))

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return len(self.data)

    dataset = SimpleDataset(10000)
    dataloader = et.ElasticDataLoader(dataset, batch_size=params["bs"], shuffle=True, num_workers=2, drop_last=True)

    optimizer = optim.SGD(net.parameters(), lr=params["lr"], momentum=0.9,
                          weight_decay=5e-4)
    lr_scheduler = MultiStepLR(optimizer, [30, 45], 0.1)

    net = et.ElasticDataParallel(net, optimizer, lr_scheduler)
    trainer = Trainer(net, optimizer, lr_scheduler)

    # Accumulator is an api of adaptdl to record training status.
    stats = et.Accumulator()
    tensorboard_dir = os.path.join(
        os.getenv("ADAPTDLCTL_TENSORBOARD_LOGDIR", "/adaptdl/tensorboard"),
        os.getenv("NNI_TRIAL_JOB_ID", "lr-adaptdl")
    )
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    with SummaryWriter(tensorboard_dir) as writer:
        for epoch in et.remaining_epochs_until(params["epochs"]):
            for inputs, targets in dataloader: 
                batch_stat = trainer.train(inputs, targets, stats)
            with stats.synchronized():
                if IS_CHIEF:
                    nni.report_intermediate_result(batch_stat["Loss"], stats)
                stats["loss_avg"] = stats["loss_sum"] / stats["total"]
                writer.add_scalar("Loss/Train", stats["loss_avg"], epoch)
        with stats.synchronized():
            if IS_CHIEF:
                nni.report_final_result(batch_stat["Loss"])


def get_params():
    parser = argparse.ArgumentParser(description='Linear Regression Training')
    parser.add_argument('--bs', default=128, type=int, help='batch size')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epochs', default=90, type=int, help='number of epochs')
    parser.add_argument('--autoscale-bsz', dest='autoscale_bsz', default=False, action='store_true', help='autoscale batchsize')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    tuner_params = nni.get_next_parameter()
    params = vars(get_params())
    params.update(tuner_params)
    main(params)
