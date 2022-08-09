import os
import subprocess
import sys

import nni
import pytorch_lightning
import pytest
import torch
import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn
import nni.retiarii.evaluator.pytorch.lightning as pl
from nni.retiarii import strategy, model_wrapper
from nni.retiarii.experiment.pytorch import RetiariiExeConfig, RetiariiExperiment
from torchvision import transforms
from torchvision.datasets import MNIST

pytestmark = pytest.mark.skipif(pytorch_lightning.__version__ < '1.0', reason='Incompatible APIs')


def nas_experiment_trial_params(rootpath):
    params = {}
    if sys.platform == 'win32':
        params['envs'] = f'set PYTHONPATH={rootpath} && '
    else:
        params['envs'] = f'PYTHONPATH={rootpath}:$PYTHONPATH'
    return params


def ensure_success(exp: RetiariiExperiment):
    # check experiment directory exists
    exp_dir = os.path.join(
        exp.config.canonical_copy().experiment_working_directory,
        exp.id
    )
    assert os.path.exists(exp_dir) and os.path.exists(os.path.join(exp_dir, 'trials'))

    # check job status
    job_stats = exp.get_job_statistics()
    if not (len(job_stats) == 1 and job_stats[0]['trialJobStatus'] == 'SUCCEEDED'):
        print('Experiment jobs did not all succeed. Status is:', job_stats, file=sys.stderr)
        print('Trying to fetch trial logs.', file=sys.stderr)

        for root, _, files in os.walk(os.path.join(exp_dir, 'trials')):
            for file in files:
                fpath = os.path.join(root, file)
                print('=' * 10 + ' ' + fpath + ' ' + '=' * 10, file=sys.stderr)
                print(open(fpath).read(), file=sys.stderr)

        raise RuntimeError('Experiment jobs did not all succeed.')


@model_wrapper
class Net(nn.Module):

    def __init__(self):
        super().__init__()
        channels = nn.ValueChoice([4, 6, 8])
        self.conv1 = nn.Conv2d(1, channels, 5)
        self.pool1 = nn.LayerChoice([
            nn.MaxPool2d((2, 2)), nn.AvgPool2d((2, 2))
        ])
        self.conv2 = nn.Conv2d(channels, 16, 5)
        self.pool2 = nn.LayerChoice([
            nn.MaxPool2d(2), nn.AvgPool2d(2), nn.Conv2d(16, 16, 2, 2)
        ])
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fcplus = nn.Linear(84, 84)
        self.shortcut = nn.InputChoice(2, 1)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        print(x.shape)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.shortcut([x, self.fcplus(x)])
        x = self.fc3(x)
        return x


def get_mnist_evaluator():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = nni.trace(MNIST)('data/mnist', download=True, train=True, transform=transform)
    train_loader = pl.DataLoader(train_dataset, 64)
    valid_dataset = nni.trace(MNIST)('data/mnist', download=True, train=False, transform=transform)
    valid_loader = pl.DataLoader(valid_dataset, 64)
    return pl.Classification(
        train_dataloader=train_loader, val_dataloaders=valid_loader,
        limit_train_batches=20,
        limit_val_batches=20,
        max_epochs=1
    )


def test_multitrial_experiment(pytestconfig):
    base_model = Net()
    evaluator = get_mnist_evaluator()
    search_strategy = strategy.Random()
    exp = RetiariiExperiment(base_model, evaluator, strategy=search_strategy)
    exp_config = RetiariiExeConfig('local')
    exp_config.trial_concurrency = 1
    exp_config.max_trial_number = 1
    exp_config._trial_command_params = nas_experiment_trial_params(pytestconfig.rootpath)
    exp.run(exp_config)
    ensure_success(exp)
    assert isinstance(exp.export_top_models()[0], dict)
    exp.stop()

def test_multitrial_experiment_resume_view(pytestconfig):
    # start a normal nas experiment
    base_model = Net()
    evaluator = get_mnist_evaluator()
    search_strategy = strategy.Random()
    exp = RetiariiExperiment(base_model, evaluator, strategy=search_strategy)
    exp_id = exp.id
    exp_config = RetiariiExeConfig('local')
    exp_config.trial_concurrency = 1
    exp_config.max_trial_number = 1
    exp_config._trial_command_params = nas_experiment_trial_params(pytestconfig.rootpath)
    exp.run(exp_config)
    ensure_success(exp)
    assert isinstance(exp.export_top_models()[0], dict)
    exp.stop()

    # resume the above nas experiment. only tested the resume logic in the python side,
    # as no more trial is executed after resume, the above experiment is already finished
    print('python api resume...')
    exp = RetiariiExperiment.resume(exp_id)
    ensure_success(exp)
    # TODO: currently `export_top_models` does not work as strategy's states are not resumed
    # assert isinstance(exp.export_top_models()[0], dict)
    exp.stop()
    # view the above experiment in non blocking mode then stop it
    print('python api view...')
    exp = RetiariiExperiment.view(exp_id, non_blocking=True)
    exp.stop()

    # the following is nnictl resume and view
    print('nnictl resume...')
    new_env = os.environ.copy()
    new_env['PYTHONPATH'] = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    proc = subprocess.run(f'nnictl resume {exp_id}', shell=True, env=new_env)
    assert proc.returncode == 0, 'resume nas experiment failed with code %d' % proc.returncode
    print('nnictl view...')
    proc = subprocess.run(f'nnictl view {exp_id}', shell=True)
    assert proc.returncode == 0, 'view nas experiment failed with code %d' % proc.returncode
    proc = subprocess.run(f'nnictl stop {exp_id}', shell=True)
    assert proc.returncode == 0, 'stop viewed nas experiment failed with code %d' % proc.returncode

def test_oneshot_experiment():
    base_model = Net()
    evaluator = get_mnist_evaluator()
    search_strategy = strategy.RandomOneShot()
    exp = RetiariiExperiment(base_model, evaluator, strategy=search_strategy)
    exp_config = RetiariiExeConfig()
    exp_config.execution_engine = 'oneshot'
    exp.run(exp_config)
    assert isinstance(exp.export_top_models()[0], dict)
