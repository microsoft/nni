# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import logging
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import nni
import utils
from utils import EarlyStopping

# set the logger format
log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    filename="networkmorphism.log",
    filemode="a",
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)
# set the logger format
logger = logging.getLogger("FashionMNIST-network-morphism")


def get_args():
    parser = argparse.ArgumentParser("FashionMNIST")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--optimizer", type=str, default="SGD", help="optimizer")
    parser.add_argument("--epoches", type=int, default=200, help="epoch limit")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--time_limit", type=int, default=0, help="gpu device id")
    parser.add_argument("--cutout", action="store_true", default=True, help="use cutout")
    parser.add_argument("--cutout_length", type=int, default=16, help="cutout length")
    parser.add_argument(
        "--model_path", type=str, default="./", help="Path to save the destination model"
    )
    args = parser.parse_args()
    return args


trainloader = None
testloader = None
net = None
criterion = None
optimizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0.0
args = get_args()


def build_graph_from_json(ir_model_path):
    """ build model from json represtation 
    """
    graph = pickle.load(open(ir_model_path, "rb"))
    logging.debug(graph.operation_history)
    model = graph.produce_torch_model()
    return model


def build_graph_from_pickle(ir_model_path):
    """ build model from pickle represtation 
    """
    graph = pickle.load(open(ir_model_path, "rb"))
    logging.debug(graph.operation_history)
    model = graph.produce_torch_model()
    return model


def parse_rev_args(receive_msg):
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer

    # Data
    logger.debug("Preparing data..")

    transform_train, transform_test = utils._data_transforms_fashion(args)

    trainset = torchvision.datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )


    # Model
    logger.debug("Building model..")
    model_path = receive_msg
    net = build_graph_from_pickle(model_path)

    net = net.to(device)
    criterion = nn.CrossEntropyLoss()

    if args.optimizer == "SGD":
        optimizer = optim.SGD(
            net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4
        )
    if args.optimizer == "Adadelta":
        optimizer = optim.Adadelta(net.parameters(), lr=args.learning_rate)
    if args.optimizer == "Adagrad":
        optimizer = optim.Adagrad(net.parameters(), lr=args.learning_rate)
    if args.optimizer == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    if args.optimizer == "Adamax":
        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    return 0


# Training
def train(epoch):
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer

    logger.debug("Epoch: %d" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        acc = 100.0 * correct / total

        logger.debug(
            "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (train_loss / (batch_idx + 1), 100.0 * correct / total, correct, total)
        )

    return acc


def test(epoch):
    global best_acc
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            acc = 100.0 * correct / total

            logger.debug(
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (test_loss / (batch_idx + 1), 100.0 * correct / total, correct, total)
            )

    acc = 100.0 * correct / total
    if acc > best_acc:
        logger.debug("Saving..")
        best_acc = acc
    return acc, best_acc


if __name__ == "__main__":
    try:
        # trial get next parameter from network morphism tuner
        RCV_CONFIG = nni.get_next_parameter()
        logger.debug(RCV_CONFIG)

        parse_rev_args(RCV_CONFIG)
        acc = 0.0
        best_acc = 0.0
        early_stop = EarlyStopping(mode="max")
        for epoch in range(args.epoches):
            train_acc = train(epoch)
            test_acc, best_acc = test(epoch)
            nni.report_intermediate_result(test_acc)
            logger.debug(test_acc)
            if early_stop.step(test_acc):
                break

        # trial report best_acc to tuner
        nni.report_final_result(best_acc)
    except Exception as exception:
        logger.exception(exception)
        raise
