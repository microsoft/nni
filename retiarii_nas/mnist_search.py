import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

import sdk
from sdk.mutators.builtin_mutators import OperatorMutator


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        #self.conv2 = nn.Conv2d(40, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 512)
        self.fc2 = nn.Linear(512, 10)
        #self.hp = retiarii.hp_choice()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        #x = torch.cat([x, x], 1)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def create_dummy_input():
    return torch.rand((64, 1, 28, 28))

class ModelTrain(sdk.Trainer):
    def __init__(self, optim_name, device='cuda', n_epochs=2):
        super(ModelTrain, self).__init__()
        self.optim_name = optim_name
        self.device = torch.device(device)
        self.n_epochs = n_epochs

    def _data_loader(self, train=True):
        dataset = MNIST("./data/mnist", train=train, download=True, transform=transforms.ToTensor())
        loader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True)
        return loader

    def train_dataloader(self):
        return self._data_loader(train=True)

    def val_dataloader(self):
        return self._data_loader(train=False)

    def configure_optimizer(self):
        if self.optim_name == 'SGD':
            return optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5)
        else:
            raise RuntimeError('Unsupported optimizer type: {}'.format(self.optim_name))

    def train_step(self, x, y, infer_y):
        assert self.model is not None
        assert self.optimizer is not None
        loss = F.cross_entropy(infer_y, y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def train(self):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_dataloader()):
            data, target = data.to(self.device), target.to(self.device)
            infer_target = self.model(data)
            self.train_step(data, target, infer_target)

    def validate(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        test_loader = self.val_dataloader()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                # sum up batch loss
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        return accuracy

# TODO: deal with `if __name__ == '__main__'`
base_model = BaseModel()
exp = sdk.create_experiment('mnist_search', base_model)
exp.specify_training(ModelTrain, 'SGD')

new_conv1_1 = nn.Conv2d(1, 20, 5, 1, bias=False)
new_conv1_2 = nn.Conv2d(1, 20, 5, 1, padding_mode='reflect')
mutator1 = OperatorMutator('conv1', [new_conv1_1, new_conv1_2])
new_conv2_1 = nn.Conv2d(20, 50, 5, 1, groups=2)
new_conv2_2 = nn.Conv2d(20, 50, 5, 1, groups=5)
mutator2 = OperatorMutator('conv2', [new_conv2_1, new_conv2_2])
exp.specify_mutators([mutator1, mutator2])
exp.specify_strategy('naive.strategy.main', 'naive.strategy.RandomSampler')
run_config = {
    'authorName': 'nas',
    'experimentName': 'nas',
    'trialConcurrency': 1,
    'maxExecDuration': '24h',
    'maxTrialNum': 999,
    'trainingServicePlatform': 'local',
    'searchSpacePath': 'empty.json',
    'useAnnotation': False
} # nni experiment config
exp.run(run_config)
