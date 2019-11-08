import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from nni.compression.torch import FilterPruner


class vgg(nn.Module):
    def __init__(self, init_weights=True):
        super(vgg, self).__init__()
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        self.cfg = cfg
        self.feature = self.make_layers(cfg, True)
        num_classes = 10
        self.classifier = nn.Sequential(
            nn.Linear(cfg[-1], 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=True):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def train(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
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
    acc = 100 * correct / len(test_loader.dataset)

    print('Loss: {}  Accuracy: {}%)\n'.format(
        test_loss, acc))
    return acc


def main():
    torch.manual_seed(0)
    device = torch.device('cuda')
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.Pad(4),
                             transforms.RandomCrop(32),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                         ])),
        batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=200, shuffle=False)

    model = vgg()
    model.to(device)

    # Train the base VGG-16 model
    print('=' * 10 + 'Train the unpruned base model' + '=' * 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 300, 0)
    for epoch in range(300):
        train(model, device, train_loader, optimizer)
        test(model, device, test_loader)
        lr_scheduler.step(epoch)
    torch.save(model.state_dict(), 'vgg16_cifar10.pth')

    # Test base model accuracy
    print('=' * 10 + 'Test on the original model' + '=' * 10)
    model.load_state_dict(torch.load('vgg16_cifar10.pth'))
    test(model, device, test_loader)
    # top1 = 93.47%

    # Pruning Configuration, in paper 'PRUNING FILTERS FOR EFFICIENT CONVNETS',
    # Conv_1, Conv_8, Conv_9, Conv_10, Conv_11, Conv_12 are pruned with 50% sparsity, as 'VGG-16-pruned-A'
    configure_list = [{
        'sparsity': 0.5,
        'op_types': ['default'],
        'op_names': ['feature.0', 'feature.24', 'feature.27', 'feature.30', 'feature.34', 'feature.37']
    }]

    # Prune model and test accuracy without fine tuning.
    print('=' * 10 + 'Test on the pruned model before fine tune' + '=' * 10)
    pruner = FilterPruner(model, configure_list)
    model = pruner.compress()
    test(model, device, test_loader)
    # top1 = 47.22%

    # Fine tune the pruned model for 40 epochs and test accuracy
    print('=' * 10 + 'Fine tuning' + '=' * 10)
    optimizer_finetune = torch.optim.SGD(model.parameters(), lr=0.004, momentum=0.9, weight_decay=1e-4)
    best_top1 = 0
    for epoch in range(40):
        pruner.update_epoch(epoch)
        print('# Epoch {} #'.format(epoch))
        train(model, device, train_loader, optimizer_finetune)
        top1 = test(model, device, test_loader)
        if top1 > best_top1:
            best_top1 = top1
            # Export the best model, 'model_path' stores state_dict of the pruned model,
            # mask_path stores mask_dict of the pruned model
            pruner.export_model(model_path='pruned_vgg16_cifar10.pth', mask_path='mask_vgg16_cifar10.pth')

    # Test the exported model
    print('=' * 10 + 'Test on the pruned model after fine tune' + '=' * 10)
    new_model = vgg()
    new_model.to(device)
    new_model.load_state_dict(torch.load('pruned_vgg16_cifar10.pth'))
    test(new_model, device, test_loader)
    # top1 = 93.18%


if __name__ == '__main__':
    main()
