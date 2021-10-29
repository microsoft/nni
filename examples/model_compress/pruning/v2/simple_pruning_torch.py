from tqdm import tqdm

import torch
from torchvision import datasets, transforms

from nni.algorithms.compression.v2.pytorch.pruning import L1NormPruner
from nni.compression.pytorch.speedup import ModelSpeedup

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

    print('\nPre-train the model:')
    for i in range(5):
        trainer(model, optimizer, criterion, i)
        evaluator(model)

    config_list = [{'op_types': ['Conv2d'], 'sparsity': 0.8}]
    pruner = L1NormPruner(model, config_list)
    _, masks = pruner.compress()

    print('\nThe accuracy with masks:')
    evaluator(model)

    pruner._unwrap_model()
    ModelSpeedup(model, dummy_input=torch.rand(10, 3, 32, 32).to(device), masks_file='simple_masks.pth').speedup_model()

    print('\nThe accuracy after speed up:')
    evaluator(model)

    # Need a new optimizer due to the modules in model will be replaced during speedup.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    print('\nFinetune the model after speed up:')
    for i in range(5):
        trainer(model, optimizer, criterion, i)
        evaluator(model)
