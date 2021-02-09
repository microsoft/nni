import random

import nni.retiarii.nn.pytorch as nn
import nni.retiarii.trainer.pytorch.lightning as pl
import torch.nn.functional as F
from nni.retiarii import blackbox_module as bm
from nni.retiarii.experiment.pytorch import RetiariiExeConfig, RetiariiExperiment
from nni.retiarii.strategies import RandomStrategy
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


class Net(nn.Module):
    def __init__(self, hidden_size):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.LayerChoice([
            nn.Linear(4*4*50, hidden_size),
            nn.Linear(4*4*50, hidden_size, bias=False)
        ])
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    base_model = Net(128)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = bm(MNIST)(root='data/mnist', train=True, download=True, transform=transform)
    test_dataset = bm(MNIST)(root='data/mnist', train=False, download=True, transform=transform)
    lightning = pl.Classification(train_dataloader=pl.DataLoader(train_dataset, batch_size=100),
                                  val_dataloaders=pl.DataLoader(test_dataset, batch_size=100),
                                  max_epochs=2)

    simple_startegy = RandomStrategy()

    exp = RetiariiExperiment(base_model, lightning, [], simple_startegy)

    exp_config = RetiariiExeConfig('local')
    exp_config.experiment_name = 'mnist_search'
    exp_config.trial_concurrency = 2
    exp_config.max_trial_number = 10
    exp_config.training_service.use_active_gpu = False

    exp.run(exp_config, 8081 + random.randint(0, 100))
