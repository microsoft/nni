from pathlib import Path

root_path = Path(__file__).parent.parent

# define the model
import torch
from torch import nn
from torch.nn import functional as F

class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# load data
from torchvision import datasets, transforms

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root_path / 'data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=128, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root_path / 'data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=1000, shuffle=True)

# define the trainer and evaluator
def trainer(model, optimizer, criterion):
    # training the model
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def evaluator(model):
    # evaluating the model accuracy and average test loss
    model.eval()
    test_loss = 0
    correct = 0
    test_dataset_length = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= test_dataset_length
    accuracy = 100. * correct / test_dataset_length
    print('Average test loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, test_dataset_length, accuracy))
