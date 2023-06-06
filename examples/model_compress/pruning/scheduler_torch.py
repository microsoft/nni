import sys
from tqdm import tqdm

import torch
from torchvision import datasets, transforms

from nni.compression.pytorch.pruning import L1NormPruner
from nni.compression.pytorch.pruning.tools import AGPTaskGenerator
from nni.compression.pytorch.pruning.basic_scheduler import PruningScheduler

from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[1] / 'models'))
from cifar10.vgg import VGG


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ]), download=True),
    batch_size=128, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=128, shuffle=False)
criterion = torch.nn.CrossEntropyLoss()

def trainer(model, optimizer, criterion, epoch):
    model.train()
    for data, target in tqdm(iterable=train_loader, desc='Epoch {}'.format(epoch)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def finetuner(model):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    for data, target in tqdm(iterable=train_loader, desc='Epoch PFs'):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def evaluator(model):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(iterable=test_loader, desc='Test'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    acc = 100 * correct / len(test_loader.dataset)
    print('Accuracy: {}%\n'.format(acc))
    return acc


if __name__ == '__main__':
    model = VGG().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # pre-train the model
    for i in range(5):
        trainer(model, optimizer, criterion, i)

    # No need to pass model and config_list to pruner during initializing when using scheduler.
    pruner = L1NormPruner(None, None)

    # you can specify the log_dir, all intermediate results and best result will save under this folder.
    # if you don't want to keep intermediate results, you can set `keep_intermediate_result=False`.
    config_list = [{'op_types': ['Conv2d'], 'sparsity': 0.8}]
    task_generator = AGPTaskGenerator(10, model, config_list, log_dir='.', keep_intermediate_result=True)

    dummy_input = torch.rand(10, 3, 32, 32).to(device)

    # if you just want to keep the final result as the best result, you can pass evaluator as None.
    # or the result with the highest score (given by evaluator) will be the best result.

    # scheduler = PruningScheduler(pruner, task_generator, finetuner=finetuner, speedup=True, dummy_input=dummy_input, evaluator=evaluator)
    scheduler = PruningScheduler(pruner, task_generator, finetuner=finetuner, speedup=True, dummy_input=dummy_input, evaluator=None, reset_weight=False)

    scheduler.compress()

    _, model, masks, _, _ = scheduler.get_best_result()
