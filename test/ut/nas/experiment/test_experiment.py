from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

import nni
from nni.nas.evaluator import FunctionalEvaluator
from nni.nas.evaluator.pytorch import Classification
from nni.nas.experiment import *
from nni.nas.experiment.config import *
from nni.nas.strategy import RegularizedEvolution, PolicyBasedRL, DARTS, Random

from ut.nas.nn.models import SimpleNet

def simple_evaluation(model, num_batches=20):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    train_dataset = MNIST('data/mnist', download=False, train=True, transform=transform)
    train_loader = DataLoader(train_dataset, 64, shuffle=True)
    valid_dataset = MNIST('data/mnist', download=False, train=False, transform=transform)
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
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    train_dataset = MNIST('data/mnist', download=False, train=True, transform=transform)
    train_loader = DataLoader(train_dataset, 64, shuffle=True)
    valid_dataset = MNIST('data/mnist', download=False, train=False, transform=transform)
    valid_loader = DataLoader(valid_dataset, 64, shuffle=True)

    evaluator = Classification(num_classes=10, limit_train_batches=10, limit_val_batches=10,
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
