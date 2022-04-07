import sys
from tqdm import tqdm

import torch
from torchvision import datasets, transforms

import nni
from nni.compression.pytorch.pruning import AutoCompressPruner

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

epoch = 0

def trainer(model, optimizer, criterion):
    global epoch
    model.train()
    for data, target in tqdm(iterable=train_loader, desc='Total Epoch {}'.format(epoch)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    epoch = epoch + 1

def finetuner(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    trainer(model, optimizer, criterion)

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
    for _ in range(10):
        trainer(model, optimizer, criterion)

    config_list = [{'op_types': ['Conv2d'], 'total_sparsity': 0.8}]
    dummy_input = torch.rand(10, 3, 32, 32).to(device)

    # make sure you have used nni.trace to wrap the optimizer class before initialize
    traced_optimizer = nni.trace(torch.optim.SGD)(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    admm_params = {
        'trainer': trainer,
        'traced_optimizer': traced_optimizer,
        'criterion': criterion,
        'iterations': 10,
        'training_epochs': 1
    }
    sa_params = {
        'evaluator': evaluator
    }
    pruner = AutoCompressPruner(model, config_list, 10, admm_params, sa_params, keep_intermediate_result=True, finetuner=finetuner)
    pruner.compress()
    _, model, masks, _, _ = pruner.get_best_result()
