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

from putils import get_parameters
from model import SearchMobileNet
from nni.nas.pytorch.proxylessnas import ProxylessNasTrainer


if __name__ == "__main__":
    parser = ArgumentParser("proxylessnas")
    # configurations of the model
    parser.add_argument("--n_cell_stages", default='4,4,4,4,4,1', type=str)
    parser.add_argument("--stride_stages", default='2,2,2,1,2,1', type=str)
    parser.add_argument("--width_stages", default='24,40,80,96,192,320', type=str)
    parser.add_argument("--bn_momentum", default=0.1, type=float)
    parser.add_argument("--bn_eps", default=1e-3, type=float)
    parser.add_argument("--dropout_rate", default=0, type=float)
    parser.add_argument("--no_decay_keys", default='bn', type=str, choices=[None, 'bn', 'bn#bias'])
    # configurations of imagenet dataset
    parser.add_argument("--data_path", default='/data/hdd3/yugzh/imagenet/', type=str)
    parser.add_argument("--train_batch_size", default=2, type=int)
    parser.add_argument("--test_batch_size", default=2, type=int)
    parser.add_argument("--n_worker", default=0, type=int)
    parser.add_argument("--resize_scale", default=0.08, type=float)
    parser.add_argument("--distort_color", default='normal', type=str, choices=['normal', 'strong', 'None'])
    #parser.add_argument("--log-frequency", default=1, type=int)
    args = parser.parse_args()

    model = SearchMobileNet(width_stages=[int(i) for i in args.width_stages.split(',')],
                            n_cell_stages=[int(i) for i in args.n_cell_stages.split(',')],
                            stride_stages=[int(i) for i in args.stride_stages.split(',')],
                            n_classes=1000,
                            dropout_rate=args.dropout_rate,
                            bn_param=(args.bn_momentum, args.bn_eps))
    print('=============================================SearchMobileNet model create done')
    model.init_model()
    print('=============================================SearchMobileNet model init done')

    # move network to GPU if available
    # data parallelism not supported yet
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        model.to(device)
    else:
        device = torch.device('cpu')

    # TODO: net info

    if args.no_decay_keys:
        keys = args.no_decay_keys
        momentum, nesterov = 0.9, True
        optimizer = torch.optim.SGD([
            {'params': get_parameters(model, keys, mode='exclude'), 'weight_decay': 4e-5},
            {'params': get_parameters(model, keys, mode='include'), 'weight_decay': 0},
        ], lr=0.05, momentum=momentum, nesterov=nesterov)
    else:
        optimizer = torch.optim.SGD(model, get_parameters(), lr=0.05, momentum=momentum, nesterov=nesterov, weight_decay=4e-5)

    print('=============================================Start to create data provider')
    # TODO: 
    data_provider = datasets.ImagenetDataProvider(save_path=args.data_path,
                                                  train_batch_size=args.train_batch_size, #256,
                                                  test_batch_size=args.test_batch_size, #500,
                                                  valid_size=None,
                                                  n_worker=args.n_worker, #32,
                                                  resize_scale=args.resize_scale,
                                                  distort_color=args.distort_color)
    print('=============================================Finish to create data provider')
    train_loader = data_provider.train
    valid_loader = data_provider.valid

    print('=============================================Start to create ProxylessNasTrainer')
    trainer = ProxylessNasTrainer(model,
                                  model_optim=optimizer,
                                  train_loader=train_loader,
                                  valid_loader=valid_loader,
                                  device=device,
                                  warmup=False)

    print('=============================================Start to train ProxylessNasTrainer')
    trainer.train()
    trainer.export()
