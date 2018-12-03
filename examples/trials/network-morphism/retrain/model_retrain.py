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


# to do
# retrain network morphism model in nni frameworks by search the best hyperparameter


import argparse
import json
import logging
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import nni
from nni.networkmorphism_tuner.graph import json_to_graph

sys.path.append("../")
import utils

# set the logger format
logger = logging.getLogger("cifar10-fine-tuning-TPE")
log_format = "%(asctime)s %(message)s"
logger.basicConfig(
    stream=sys.stdout, level=logging.INFO, format=log_format, datefmt="%m/%d %I:%M:%S %p"
)


def get_args():
    parser = argparse.ArgumentParser("cifar10")
    parser.add_argument("--cutout", action="store_true", default=True, help="use cutout")
    parser.add_argument("--cutout_length", type=int, default=6, help="cutout length")
    args = parser.parse_args()
    return args


trainloader = None
testloader = None
net = None
criterion = None
optimizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0.0
epoches = 0
best_model_json = dict()
args = get_args()


def build_graph_from_json(ir_model_json):
    """build model from json representation
    """
    graph = json_to_graph(ir_model_json)
    logging.debug(graph.operation_history)
    model = graph.produce_torch_model()
    return model


def prepare(msg):
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer
    global best_model_path

    # Loading Data
    logger.info("Preparing data..")

    transform_train, transform_test = utils._data_transforms_cifar10(args)

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=msg["batch_size"], shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=msg["batch_size"], shuffle=False, num_workers=2
    )

    # Model
    logger.info("Building model..")
    net = build_graph_from_json(best_model_json)

    net = net.to(device)

    criterion = nn.CrossEntropyLoss()

    if msg["optimizer"] == "SGD":
        optimizer = optim.SGD(
            net.parameters(),
            lr=msg["learning_rate"],
            momentum=0.9,
            weight_decay=msg["weight_decay"],
        )
    if msg["optimizer"] == "Adadelta":
        optimizer = optim.Adadelta(
            net.parameters(), lr=msg["learning_rate"], weight_decay=msg["weight_decay"]
        )
    if msg["optimizer"] == "Adagrad":
        optimizer = optim.Adagrad(
            net.parameters(), lr=msg["learning_rate"], weight_decay=msg["weight_decay"]
        )
    if msg["optimizer"] == "Adam":
        optimizer = optim.Adam(
            net.parameters(), lr=msg["learning_rate"], weight_decay=msg["weight_decay"]
        )
    if msg["optimizer"] == "Adamax":
        optimizer = optim.Adamax(
            net.parameters(), lr=msg["learning_rate"], weight_decay=msg["weight_decay"]
        )
    if msg["optimizer"] == "RMSprop":
        optimizer = optim.RMSprop(
            net.parameters(), lr=msg["learning_rate"], weight_decay=msg["weight_decay"]
        )


# Training
def train(epoch):
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer

    logger.info("Epoch: %d" % epoch)
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

    # Save checkpoint.
    acc = 100.0 * correct / total
    if acc > best_acc:
        logger.info("Saving..")
        best_acc = acc
    return acc, best_acc


if __name__ == "__main__":
    try:
        args = get_args()
        # trial get next parameter from tuner
        RCV_CONFIG = nni.get_next_parameter()
        logger.debug(RCV_CONFIG)

        prepare(RCV_CONFIG)
        acc = 0.0
        best_acc = 0.0
        for epoch in range(epoches):
            train(epoch)
            acc, best_acc = test(epoch)
            nni.report_intermediate_result(acc)

        # trial report best_acc to tuner
        nni.report_final_result(best_acc)
    except Exception as exception:
        logger.exception(exception)
        raise
