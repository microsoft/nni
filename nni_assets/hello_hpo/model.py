"""
Run main.py to start.

This script is modified from PyTorch quickstart:
https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
"""

import nni
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Get optimized hyperparameters
params = {'features': 512, 'lr': 0.001, 'momentum': 0}
optimized_params = nni.get_next_parameter()
params.update(optimized_params)

# Load dataset
training_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=ToTensor())
train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# Build model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, params['features']),
    nn.ReLU(),
    nn.Linear(params['features'], params['features']),
    nn.ReLU(),
    nn.Linear(params['features'], 10)
).to(device)

# Training functions
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], momentum=params['momentum'])

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(dataloader, model, loss_fn):
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    return correct / len(dataloader.dataset)

# Train the model
epochs = 5
for t in range(epochs):
    train(train_dataloader, model, loss_fn, optimizer)
    accuracy = test(test_dataloader, model, loss_fn)
    nni.report_intermediate_result(accuracy)
nni.report_final_result(accuracy)
