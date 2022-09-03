import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import nni.nas.nn.pytorch

import torch


class _model(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = stem()
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(out_features=256, in_features=1024)
        self.fc2 = torch.nn.Linear(out_features=10, in_features=256)
        self.softmax = torch.nn.Softmax()
        self._mapping_ = {'stem': None, 'flatten': None, 'fc1': None, 'fc2': None, 'softmax': None}

    def forward(self, image):
        stem = self.stem(image)
        flatten = self.flatten(stem)
        fc1 = self.fc1(flatten)
        fc2 = self.fc2(fc1)
        softmax = self.softmax(fc2)
        return softmax



class stem(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(out_channels=32, in_channels=1, kernel_size=5)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(out_channels=64, in_channels=32, kernel_size=5)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2)
        self._mapping_ = {'conv1': None, 'pool1': None, 'conv2': None, 'pool2': None}

    def forward(self, *_inputs):
        conv1 = self.conv1(_inputs[0])
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        return pool2
