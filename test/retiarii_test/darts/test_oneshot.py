import json
import numpy as np
import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from torchvision import transforms
from torchvision.datasets import CIFAR10

from nni.retiarii.experiment import RetiariiExperiment, RetiariiExeConfig
from nni.retiarii.strategies import TPEStrategy
from nni.retiarii.trainer.pytorch import DartsTrainer

from darts_model import CNN

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img


def get_dataset(cls, cutout_length=0):
    MEAN = [0.49139968, 0.48215827, 0.44653124]
    STD = [0.24703233, 0.24348505, 0.26158768]
    transf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]
    cutout = []
    if cutout_length > 0:
        cutout.append(Cutout(cutout_length))

    train_transform = transforms.Compose(transf + normalize + cutout)
    valid_transform = transforms.Compose(normalize)

    if cls == "cifar10":
        dataset_train = CIFAR10(root="./data/cifar10", train=True, download=True, transform=train_transform)
        dataset_valid = CIFAR10(root="./data/cifar10", train=False, download=True, transform=valid_transform)
    else:
        raise NotImplementedError
    return dataset_train, dataset_valid

def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = dict()
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res["acc{}".format(k)] = correct_k.mul_(1.0 / batch_size).item()
    return res

if __name__ == '__main__':
    base_model = CNN(32, 3, 16, 10, 8)

    dataset_train, dataset_valid = get_dataset("cifar10")
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(base_model.parameters(), 0.025, momentum=0.9, weight_decay=3.0E-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 50, eta_min=0.001)
    trainer = DartsTrainer(
        model=base_model,
        loss=criterion,
        metrics=lambda output, target: accuracy(output, target, topk=(1,)),
        optimizer=optim,
        num_epochs=50,
        dataset=dataset_train,
        batch_size=32,
        log_frequency=10,
        unrolled=False
    )

    exp = RetiariiExperiment(base_model, trainer)
    exp.run()
