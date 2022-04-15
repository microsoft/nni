import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

logger = logging.getLogger('mnist_AutoML')

class WrapLinear(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc2 = nn.Linear(hidden_size, 32)

    def forward(self, x):
        return self.fc2(x)

class Net(nn.Module):
    def __init__(self, hidden_size):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, hidden_size)
        #self.fc2 = nn.Linear(hidden_size, 10)
        self.wfc2 = WrapLinear(hidden_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.wfc2(x)
        return F.log_softmax(x, dim=1)

from sparta import SpartaModel
model = Net(128)
wraplinear = model.wfc2
linear = wraplinear.fc2
tesa = torch.ones_like(linear.weight, dtype=torch.int8)
tesa[0][0] = 0
setattr(linear, 'weight_tesa', tesa)
in_tesa = torch.ones(32, 128, dtype=torch.int8)
setattr(linear, 'input_tesa', in_tesa)
out_tesa = torch.ones(32, 10, dtype=torch.int8)
setattr(linear, 'output_tesa', out_tesa)

opt_model = SpartaModel(model)
data = torch.ones(1, 1, 28, 28)
out = opt_model(data)
print(out)
