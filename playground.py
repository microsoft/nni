import pytorch_lightning as pl

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

import json_tricks as jsont


dataset = CIFAR10(root='test/retiarii_test/darts/data/cifar10', download=False)
dataloader = DataLoader(dataset, batch_size=10)

print(jsont.dumps(dataloader))
