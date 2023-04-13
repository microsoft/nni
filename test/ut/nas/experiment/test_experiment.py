import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset

import nni
from nni.nas.evaluator import FunctionalEvaluator
from nni.nas.evaluator.pytorch import Classification
from nni.nas.experiment import *
from nni.nas.experiment.config import *
from nni.nas.strategy import RegularizedEvolution, PolicyBasedRL, DARTS, Random

from ut.nas.nn.models import SimpleNet

class RandomMnistDataset(Dataset):
    def __init__(self, length):
        self.len = length
        self.inputs = torch.randn(length, 1, 28, 28)
        self.targets = torch.randint(10, (length,))

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return self.len

def simple_evaluation(model, num_batches=20):
    train_dataset = RandomMnistDataset(1000)
    train_loader = DataLoader(train_dataset, 64, shuffle=True)
    valid_dataset = RandomMnistDataset(200)
    valid_loader = DataLoader(valid_dataset, 64, shuffle=True)

    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for _, (x, y) in zip(range(num_batches), train_loader):
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    accurate, total = 0, 0
    for _, (x, y) in zip(range(num_batches), valid_loader):
        y_hat = model(x)
        accurate += (y_hat.argmax(1) == y).sum().item()
        total += y.shape[0]
    nni.report_final_result(accurate / total)

def test_experiment_sanity():
    model_space = SimpleNet()
    evaluator = FunctionalEvaluator(simple_evaluation)
    strategy = RegularizedEvolution(population_size=2, sample_size=2)

    exp = NasExperiment(model_space, evaluator, strategy)
    exp.config.max_trial_number = 3
    exp.run()
    assert isinstance(exp.export_top_models(formatter='instance')[0], SimpleNet)
    exp.stop()

def test_oneshot_sanity():
    model_space = SimpleNet()
    train_dataset = RandomMnistDataset(1000)
    train_loader = DataLoader(train_dataset, 64, shuffle=True)
    valid_dataset = RandomMnistDataset(200)
    valid_loader = DataLoader(valid_dataset, 64, shuffle=True)

    evaluator = Classification(num_classes=10,
                               train_dataloaders=train_loader,
                               val_dataloaders=valid_loader,
                               max_epochs=2)
    strategy = DARTS()

    exp = NasExperiment(model_space, evaluator, strategy)
    exp.run()
    assert isinstance(exp.export_top_models(formatter='dict')[0], dict)

def test_experiment_resume():
    model_space = SimpleNet()
    evaluator = FunctionalEvaluator(simple_evaluation)
    strategy = RegularizedEvolution(population_size=2, sample_size=2)

    config = NasExperimentConfig('sequential', 'raw')
    config.max_trial_number = 3
    exp = NasExperiment(model_space, evaluator, strategy, config)
    exp.run_or_resume(debug=True)
    exp.stop()

    strategy2 = RegularizedEvolution(population_size=2, sample_size=2)
    config = NasExperimentConfig('sequential', 'raw')
    config.max_trial_number = 5
    exp = NasExperiment(model_space, evaluator, strategy2, config, id=exp.id)
    exp.run_or_resume(debug=True)
    exp.stop()
