import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class _model(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = stem()
        
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 10)
        

    def forward(self, image):
        stem = self.stem(image)
        flatten = stem.view(stem.size(0), -1)
        fc1 = self.fc1(flatten)
        fc2 = self.fc2(fc1)
        softmax = F.softmax(fc2, -1)
        return softmax



class stem(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(out_channels=32, in_channels=1, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(out_channels=64, in_channels=32, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

    def forward(self, *_inputs):
        conv1 = self.conv1(_inputs[0])
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        return pool2
