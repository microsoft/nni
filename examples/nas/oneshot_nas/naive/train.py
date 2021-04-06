import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from nni.nas.pytorch.mutables import LayerChoice, InputChoice
from nni.algorithms.nas.pytorch.darts import DartsTrainer


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = LayerChoice([nn.Conv2d(3, 6, 3, padding=1), nn.Conv2d(3, 6, 5, padding=2)])
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = LayerChoice([nn.Conv2d(6, 16, 3, padding=1), nn.Conv2d(6, 16, 5, padding=2)])
        self.conv3 = nn.Conv2d(16, 16, 1)

        self.skipconnect = InputChoice(n_candidates=1)
        self.bn = nn.BatchNorm2d(16)

        self.gap = nn.AdaptiveAvgPool2d(4)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        bs = x.size(0)

        x = self.pool(F.relu(self.conv1(x)))
        x0 = F.relu(self.conv2(x))
        x1 = F.relu(self.conv3(x0))

        x0 = self.skipconnect([x0])
        if x0 is not None:
            x1 += x0
        x = self.pool(self.bn(x1))

        x = self.gap(x).view(bs, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def accuracy(output, target):
    batch_size = target.size(0)
    _, predicted = torch.max(output.data, 1)
    return {"acc1": (predicted == target).sum().item() / batch_size}


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    dataset_valid = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    trainer = DartsTrainer(net,
                           loss=criterion,
                           metrics=accuracy,
                           optimizer=optimizer,
                           num_epochs=2,
                           dataset_train=dataset_train,
                           dataset_valid=dataset_valid,
                           batch_size=64,
                           log_frequency=10)
    trainer.enable_visualization()
    trainer.train()
    trainer.export("checkpoint.json")
