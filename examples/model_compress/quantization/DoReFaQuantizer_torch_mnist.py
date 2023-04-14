import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from nni.compression.pytorch.quantization import DoReFaQuantizer

import sys
sys.path.append('../models')
from mnist.naive import NaiveModel


def train(model, quantizer, device, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('{:2.0f}%  Loss {}'.format(100 * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('Loss: {}  Accuracy: {}%)\n'.format(
        test_loss, 100 * correct / len(test_loader.dataset)))

def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=trans),
        batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=trans),
        batch_size=1000, shuffle=True)

    model = NaiveModel()
    model = model.to(device)
    configure_list = [{
        'quant_types': ['weight'],
        'quant_bits': {
            'weight': 8,
        }, # you can just use `int` here because all `quan_types` share same bits length, see config for `ReLu6` below.
        'op_types':['Conv2d', 'Linear']
    }]
    quantizer = DoReFaQuantizer(model, configure_list)
    quantizer.compress()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
    for epoch in range(10):
        print('# Epoch {} #'.format(epoch))
        train(model, quantizer, device, train_loader, optimizer)
        test(model, device, test_loader)


if __name__ == '__main__':
    main()
