from tqdm import tqdm

import torch
from torchvision import datasets, transforms

from nni.algorithms.compression.v2.pytorch.pruning import AGPPruner

from examples.model_compress.models.cifar10.vgg import VGG


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

    config_list = [{'op_types': ['Conv2d'], 'sparsity': 0.8}]
    dummy_input = torch.rand(10, 3, 32, 32).to(device)

    # if you just want to keep the final result as the best result, you can pass evaluator as None.
    # or the result with the highest score (given by evaluator) will be the best result.

    # pruner = AGPPruner(model, config_list, 'l1', 10, finetuner=finetuner, speed_up=True, dummy_input=dummy_input, evaluator=evaluator)
    pruner = AGPPruner(model, config_list, 'l1', 10, finetuner=finetuner, speed_up=True, dummy_input=dummy_input, evaluator=None)
    pruner.compress()
    _, model, masks, _, _ = pruner.get_best_result()
