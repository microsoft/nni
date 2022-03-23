"""
Port PyTorch Quickstart to NNI
==============================
This is a modified version of `PyTorch quickstart`_.

It can be run directly and will have the exact same result as original version.

Furthermore, it enables the ability of auto tuning with an NNI *experiment*, which will be detailed later.

It is recommended to run this script directly first to verify the environment.

There are 2 key differences from the original version:

1. In `Get optimized hyperparameters`_ part, it receives generated hyperparameters.
2. In `Train model and report accuracy`_ part, it reports accuracy metrics to NNI.

.. _PyTorch quickstart: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
"""

# %%
import nni
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# %%
# Hyperparameters to be tuned
# ---------------------------
# These are the hyperparameters that will be tuned.
params = {
    'features': 512,
    'lr': 0.001,
    'momentum': 0,
}

# %%
# Get optimized hyperparameters
# -----------------------------
# If run directly, :func:`nni.get_next_parameter` is a no-op and returns an empty dict.
# But with an NNI *experiment*, it will receive optimized hyperparameters from tuning algorithm.
optimized_params = nni.get_next_parameter()
params.update(optimized_params)
print(params)

# %%
# Load dataset
# ------------
training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# %%
# Build model with hyperparameters
# --------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, params['features']),
            nn.ReLU(),
            nn.Linear(params['features'], params['features']),
            nn.ReLU(),
            nn.Linear(params['features'], 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], momentum=params['momentum'])

# %%
# Define train and test
# ---------------------
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    return correct

# %%
# Train model and report accuracy
# -------------------------------
# Report accuracy metrics to NNI so the tuning algorithm can suggest better hyperparameters.
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    accuracy = test(test_dataloader, model, loss_fn)
    nni.report_intermediate_result(accuracy)
nni.report_final_result(accuracy)
