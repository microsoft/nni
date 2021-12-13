import random

import nni
import torch
import torch.nn as nn
import torch.nn.functional as F
import nni.retiarii.strategy as strategy
import nni.retiarii.evaluator.pytorch.lightning as pl
from nni.retiarii import model_wrapper
from nni.retiarii.nn.pytorch import LayerChoice, ValueChoice
from nni.retiarii.evaluator import FunctionalEvaluator
from nni.retiarii.experiment.pytorch import RetiariiExeConfig, RetiariiExperiment, debug_mutated_model
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


@model_wrapper
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = LayerChoice([
            nn.Conv2d(32, 64, 3, 1),
            DepthwiseSeparableConv(32, 64)
        ])
        self.dropout1 = nn.Dropout(nn.ValueChoice([0.25, 0.5, 0.75]))
        self.dropout2 = nn.Dropout(0.5)
        feature = ValueChoice([64, 128, 256])
        self.fc1 = nn.Linear(9216, feature)
        self.fc2 = nn.Linear(feature, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(self.conv2(x), 2)
        x = torch.flatten(self.dropout1(x), 1)
        x = self.fc2(self.dropout2(F.relu(self.fc1(x))))
        return x


def train_epoch(model, device, train_loader, optimizer, epoch):
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test_epoch(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset), accuracy))

    return accuracy


def evaluate_model(model_cls):
    # NOTE: "model_cls" is a class, need to instantiate
    model = model_cls()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(MNIST('data/mnist', download=True, transform=transf), batch_size=64, shuffle=True)
    test_loader = DataLoader(MNIST('data/mnist', download=True, train=False, transform=transf), batch_size=64)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    for epoch in range(3):
        train_epoch(model, device, train_loader, optimizer, epoch)
        accuracy = test_epoch(model, device, test_loader)
        # NOTE: call report intermediate result. Result can be float or dict
        nni.report_intermediate_result(accuracy)

    # NOTE: report final test result
    nni.report_final_result(accuracy)


if __name__ == '__main__':
    base_model = Net()

    search_strategy = strategy.Random()
    model_evaluator = FunctionalEvaluator(evaluate_model)

    exp = RetiariiExperiment(base_model, model_evaluator, [], search_strategy)

    exp_config = RetiariiExeConfig('local')
    exp_config.experiment_name = 'mnist_search'
    exp_config.trial_concurrency = 2
    exp_config.max_trial_number = 20
    exp_config.training_service.use_active_gpu = False
    export_formatter = 'dict'

    # uncomment this for graph-based execution engine
    # exp_config.execution_engine = 'base'
    # export_formatter = 'code'

    exp.run(exp_config, 8081 + random.randint(0, 100))
    print('Final model:')
    for model_code in exp.export_top_models(formatter=export_formatter):
        print(model_code)
