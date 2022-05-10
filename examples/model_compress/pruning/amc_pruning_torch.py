import sys
from tqdm import tqdm

import torch
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import MultiStepLR

from nni.compression.pytorch.pruning import AMCPruner
from nni.compression.pytorch.utils import count_flops_params

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
    # model = MobileNetV2(n_class=10).to(device)
    model = VGG().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    for i in range(100):
        trainer(model, optimizer, criterion, i)
    pre_best_acc = evaluator(model)

    dummy_input = torch.rand(10, 3, 32, 32).to(device)
    pre_flops, pre_params, _ = count_flops_params(model, dummy_input)

    config_list = [{'op_types': ['Conv2d'], 'total_sparsity': 0.5, 'max_sparsity_per_layer': 0.8}]

    # if you just want to keep the final result as the best result, you can pass evaluator as None.
    # or the result with the highest score (given by evaluator) will be the best result.
    ddpg_params = {'hidden1': 300, 'hidden2': 300, 'lr_c': 1e-3, 'lr_a': 1e-4, 'warmup': 100, 'discount': 1., 'bsize': 64,
                   'rmsize': 100, 'window_length': 1, 'tau': 0.01, 'init_delta': 0.5, 'delta_decay': 0.99, 'max_episode_length': 1e9, 'epsilon': 50000}
    pruner = AMCPruner(400, model, config_list, dummy_input, evaluator, finetuner=finetuner, ddpg_params=ddpg_params, target='flops')
    pruner.compress()
    _, model, masks, best_acc, _ = pruner.get_best_result()
    flops, params, _ = count_flops_params(model, dummy_input)
    print(f'Pretrained model FLOPs {pre_flops/1e6:.2f} M, #Params: {pre_params/1e6:.2f}M, Accuracy: {pre_best_acc: .2f}%')
    print(f'Finetuned model FLOPs {flops/1e6:.2f} M, #Params: {params/1e6:.2f}M, Accuracy: {best_acc: .2f}%')
