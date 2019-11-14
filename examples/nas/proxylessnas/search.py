# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge,
# to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from argparse import ArgumentParser

import datasets
import torch
import torch.nn as nn

from model import *
from nni.nas.pytorch.proxylessnas import ProxylessNasTrainer

def get_parameters(model, keys=None, mode='include'):
    if keys is None:
        for name, param in model.named_parameters():
            yield param
    elif mode == 'include':
        for name, param in model.named_parameters():
            flag = False
            for key in keys:
                if key in name:
                    flag = True
                    break
            if flag:
                yield param
    elif mode == 'exclude':
        for name, param in model.named_parameters():
            flag = True
            for key in keys:
                if key in name:
                    flag = False
                    break
            if flag:
                yield param
    else:
        raise ValueError('do not support: %s' % mode)


if __name__ == "__main__":
    parser = ArgumentParser("proxylessnas")
    parser.add_argument("--layers", default=4, type=int)
    parser.add_argument("--nodes", default=2, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--log-frequency", default=1, type=int)
    args = parser.parse_args()

    #dataset_train, dataset_valid = datasets.get_dataset("cifar10")

    model = SearchMobileNet()
    print('=============================================SearchMobileNet model create done')
    model.init_model()
    print('=============================================SearchMobileNet model init done')

    # move network to GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        #self.net = torch.nn.DataParallel(self.net)
        model.to(device)
        #cudnn.benchmark = True
    else:
        raise ValueError
        # self.device = torch.device('cpu')

    # TODO: net info

    # TODO: removed decay_key
    no_decay_keys = True
    if no_decay_keys:
        keys = ['bn']
        momentum, nesterov = 0.9, True
        optimizer = torch.optim.SGD([
            {'params': get_parameters(model, keys, mode='exclude'), 'weight_decay': 4e-5},
            {'params': get_parameters(model, keys, mode='include'), 'weight_decay': 0},
        ], lr=0.05, momentum=momentum, nesterov=nesterov)
    else:
        optimizer = torch.optim.SGD(model, get_parameters(), lr=0.05, momentum=momentum, nesterov=nesterov, weight_decay=4e-5)

    #n_epochs = 50
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, n_epochs, eta_min=0.001)

    print('=============================================Start to create data provider')
    # TODO: 
    data_provider = datasets.ImagenetDataProvider(save_path='/data/hdd3/yugzh/imagenet/',
                                         train_batch_size=256,
                                         test_batch_size=500,
                                         valid_size=None,
                                         n_worker=0, #32,
                                         resize_scale=0.08,
                                         distort_color='normal')
    print('=============================================Finish to create data provider')
    train_loader = data_provider.train
    valid_loader = data_provider.valid

    print('=============================================Start to create ProxylessNasTrainer')
    trainer = ProxylessNasTrainer(model,
                                  model_optim=optimizer,
                                  train_loader=train_loader,
                                  valid_loader=valid_loader,
                                  device=device)

    print('=============================================Start to train ProxylessNasTrainer')
    trainer.train()
    trainer.export()
