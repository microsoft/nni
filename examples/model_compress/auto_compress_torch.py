# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import logging
import torch

from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from pruning.models.mnist.lenet import LeNet

from nni.algorithms.compression.pytorch.autocompressor import AutoCompressor

_logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, device, train_loader, test_loader, epochs, gamma, log_interval=10):
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.gamma = gamma

        self.log_interval = log_interval

    def pretrain(self, model, optimizer):
        print('start pre-training')
        scheduler = StepLR(optimizer, step_size=1, gamma=self.gamma)
        for epoch in range(1, self.epochs + 1):
            self.__train(model, optimizer, epoch)
            self.__test(model)
            scheduler.step()

    def finetune(self, model, optimizer, pruner):
        best_top1 = 0
        for epoch in range(1, self.epochs + 1):
            self.__train(model, optimizer, epoch)
            top1 = self.__test(model)

            if top1 > best_top1:
                best_top1 = top1
                pruner.save('multicompressor_dict.pkl')

        return 'multicompressor_dict.pkl'

    def __train(self, model, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()))

    def __test(self, model):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        acc = 100 * correct / len(self.test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset), acc))

        return acc


def main(args):
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset1 = datasets.MNIST('data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    epochs = args.epochs
    log_interval = args.log_interval

    trainer = Trainer(device, train_loader, test_loader, epochs, log_interval)

    model = LeNet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    trainer.pretrain(model, optimizer)

    torch.save(model.state_dict(), "pretrain_mnist_lenet.pt")

    print('start pruning')
    optimizer_finetune = torch.optim.SGD(model.parameters(), lr=0.01)

    pruner = AutoCompressor(model, optimizer=optimizer_finetune, trainer=trainer)

    pruner.run(input_shape=[10, 1, 28, 28], device=device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example for model comporession')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--sparsity', type=float, default=0.5,
                        help='target overall target sparsity')
    args = parser.parse_args()

    main(args)
