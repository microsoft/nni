"""
Pruning Quickstart
==================
Model pruning usually have following modes:
#. Pre-training a model -> Pruning the model -> Fine-tuning the model
#. Pruning the model aware training -> Fine-tuning the model
#. Pruning the model -> Pre-training the compact model
NNI supports the above three modes and mainly focuses on the pruning stage.
Follow this tutorial for a quick look at how to use NNI to pruning a model with mode 1.
"""

# %%
# Pre-training Model
# ------------------
#
# Initializing a naive model and optimizer

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD

class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.relu1 = nn.ReLU6()
        self.relu2 = nn.ReLU6()
        self.relu3 = nn.ReLU6()
        self.max_pool1 = nn.MaxPool2d(2, 2)
        self.max_pool2 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.max_pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.max_pool2(x)
        x = x.view(-1, x.size()[1:].numel())
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = TorchModel().to(device)
optimizer = SGD(model.parameters(), 1e-2)

# %%
# Prepare data

from torchvision import datasets, transforms

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=1000, shuffle=True)

# %%
# Pre-training

def trainer(model, optimizer, epoch):
    # training the model
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

    # evaluating the model accuracy
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('After Epoch {} Training: Average test loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        epoch, test_loss, correct, len(test_loader.dataset), accuracy))

for epoch in range(3):
    trainer(model, optimizer, epoch)

# %%
# Pruning Model
# -------------
#
# Using L1NormPruner pruning the model and generating the masks.
# Usually, pruners require original model and ``config_list`` as parameters.
# Detailed about how to write ``config_list`` please refer ...

from nni.algorithms.compression.v2.pytorch.pruning import L1NormPruner

config_list = [{
    'sparsity': 0.5,
    'op_types': ['Linear', 'Conv2d']
}, {
    'exclude': True,
    'op_names': ['fc2']
}]
pruner = L1NormPruner(model, config_list)
_, masks = pruner.compress()

# %%
# Speed up the original model with masks.

from nni.compression.pytorch.speedup import ModelSpeedup

pruner._unwrap_model()
ModelSpeedup(model, torch.rand(3, 1, 28, 28).to(device), masks).speedup_model()

print(model)

# %%
# Fine-tuning Compacted Model
# ---------------------------
# Note that if the model has been sped up, you need to re-initialize a new optimizer for fine-tuning.
# Because speed up will replace the masked big layers with dense small ones.

optimizer = SGD(model.parameters(), 1e-2)
for epoch in range(3):
    trainer(model, optimizer, epoch)
