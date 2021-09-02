import random

import torch
import nni.retiarii.nn.pytorch as nn
import nni.retiarii.strategy as strategy
import nni.retiarii.evaluator.pytorch.lightning as pl
import torch.nn.functional as F
from nni.retiarii import serialize, model_wrapper
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
        self.conv2 = nn.LayerChoice([
            nn.Conv2d(32, 64, 3, 1),
            DepthwiseSeparableConv(32, 64)
        ])
        self.dropout1 = nn.Dropout(nn.ValueChoice([0.25, 0.5, 0.75]))
        self.dropout2 = nn.Dropout(0.5)
        feature = nn.ValueChoice([64, 128, 256])
        self.fc1 = nn.Linear(9216, feature)
        self.fc2 = nn.Linear(feature, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(self.conv2(x), 2)
        x = torch.flatten(self.dropout1(x), 1)
        x = self.fc2(self.dropout2(F.relu(self.fc1(x))))
        output = F.log_softmax(x, dim=1)
        return output

if __name__ == '__main__':
    base_model = Net()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = serialize(MNIST, root='data/mnist', train=True, download=True, transform=transform)
    test_dataset = serialize(MNIST, root='data/mnist', train=False, download=True, transform=transform)
    trainer = pl.Classification(train_dataloader=pl.DataLoader(train_dataset, batch_size=100),
                                val_dataloaders=pl.DataLoader(test_dataset, batch_size=100),
                                max_epochs=2)

    simple_strategy = strategy.Random()

    exp = RetiariiExperiment(base_model, trainer, [], simple_strategy)

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