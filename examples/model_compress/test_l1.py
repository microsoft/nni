from __future__ import print_function

import argparse
import os
import json
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms, models

from models.mnist.lenet import LeNet
from nni.compression.torch import SimulatedAnnealingPruner
from nni.compression.torch import L1FilterPruner


def train(args, model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, criterion, val_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            # test_loss += F.nll_loss(output, target, reduction='sum').item()
            test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset), 100. * accuracy))

    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--pruning-mode', type=str, default='channel', metavar='P',
                        help='pruning mode, channel or fine_grained')
    parser.add_argument('--sparsity', type=float, default=0.1, metavar='S',
                        help='overall target sparsity')

    parser.add_argument('--dataset', type=str, default='mnist', metavar='DS',
                        help='dataset to use, mnist or imagenet (default MNIST)')
    parser.add_argument('--data-dir', type=str,
                        default='/datasets/', metavar='F')
    parser.add_argument('--fine-tune', type=bool, default=True, metavar='F',
                        help='Whether to fine-tune the pruned model')
    parser.add_argument('--fine-tune-epochs', type=int, default=10, metavar='N',
                        help='epochs to fine tune')
    parser.add_argument('--experiment-data-dir', type=str,
                        default='./', help='For saving experiment data')

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',  # TODO:14
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    args = parser.parse_args()

    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {
    }

    if args.dataset == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(args.data_dir, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.MNIST(args.data_dir, train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
        criterion = nn.NLLLoss()
    elif args.dataset == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(args.data_dir, 'train'),
                                 transform=transforms.Compose([
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     normalize,
                                 ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(args.data_dir, 'val'),
                                 transform=transforms.Compose([
                                     transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     normalize,
                                 ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
        criterion = nn.CrossEntropyLoss()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'mnist':
        model = LeNet().to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(args.epochs):
            train(args, model, device, train_loader,
                  criterion, optimizer, epoch)
            scheduler.step()
    elif args.dataset == 'imagenet':
        model = models.mobilenet_v2(pretrained=True).to(device)

    def evaluator(model):
        return test(model, device, criterion, val_loader)

    result = {}

    # evaluation_result = evaluator(model)
    # print('Evaluation result (original model): %s' % evaluation_result)
    # result['original'] = evaluation_result

    # config_list = [{
    #     'sparsity': args.sparsity,
    #     # module types to prune, only "Conv2d" supported for channel pruning
    #     'op_types': ['Conv2d']
    # }]

    config_list = [{'sparsity': 0.0, 'op_types': ['Conv2d'], 'op_names': ['features.1.conv.0.0']},
                   {'sparsity': 0.0, 'op_types': ['Conv2d'], 'op_names': ['features.1.conv.1']}, {'sparsity': 0.0, 'op_types': [
                       'Conv2d'], 'op_names': ['features.0.0']}, {'sparsity': 0.0, 'op_types': ['Conv2d'], 'op_names': ['features.2.conv.1.0']},
                   {'sparsity': 0.0, 'op_types': ['Conv2d'], 'op_names': ['features.3.conv.1.0']}, {
                       'sparsity': 0.0, 'op_types': ['Conv2d'], 'op_names': ['features.4.conv.1.0']},
                   {'sparsity': 0.0, 'op_types': ['Conv2d'], 'op_names': ['features.2.conv.0.0']}, {
                       'sparsity': 0.0, 'op_types': ['Conv2d'], 'op_names': ['features.5.conv.1.0']},
                   {'sparsity': 0.0, 'op_types': [
                       'Conv2d'], 'op_names': ['features.6.conv.1.0']},
                   {'sparsity': 0.0, 'op_types': [
                       'Conv2d'], 'op_names': ['features.7.conv.1.0']},
                   {'sparsity': 0.0, 'op_types': [
                       'Conv2d'], 'op_names': ['features.2.conv.2']},
                   {'sparsity': 0.0, 'op_types': [
                       'Conv2d'], 'op_names': ['features.3.conv.0.0']},
                   {'sparsity': 0.02959186005198498, 'op_types': ['Conv2d'], 'op_names': ['features.3.conv.2']}, {
                       'sparsity': 0.03447521909615123, 'op_types': ['Conv2d'], 'op_names': ['features.4.conv.0.0']},
                   {'sparsity': 0.04305503722502199, 'op_types': ['Conv2d'], 'op_names': ['features.8.conv.1.0']}, {
                       'sparsity': 0.057466011766346456, 'op_types': ['Conv2d'], 'op_names': ['features.9.conv.1.0']},
                   {'sparsity': 0.05762489326362369, 'op_types': [
                       'Conv2d'], 'op_names': ['features.10.conv.1.0']},
                   {'sparsity': 0.0586464514197868, 'op_types': ['Conv2d'], 'op_names': ['features.11.conv.1.0']}, {
                       'sparsity': 0.06110641726690995, 'op_types': ['Conv2d'], 'op_names': ['features.4.conv.2']},
                   {'sparsity': 0.062105850854006245, 'op_types': [
                       'Conv2d'], 'op_names': ['features.12.conv.1.0']},
                   {'sparsity': 0.06511706557908171, 'op_types': [
                       'Conv2d'], 'op_names': ['features.13.conv.1.0']},
                   {'sparsity': 0.07150621066727708, 'op_types': [
                       'Conv2d'], 'op_names': ['features.14.conv.1.0']},
                   {'sparsity': 0.07354172657833935, 'op_types': [
                       'Conv2d'], 'op_names': ['features.5.conv.0.0']},
                   {'sparsity': 0.07621537520107129, 'op_types': [
                       'Conv2d'], 'op_names': ['features.5.conv.2']},
                   {'sparsity': 0.07875377087122815, 'op_types': [
                       'Conv2d'], 'op_names': ['features.6.conv.0.0']},
                   {'sparsity': 0.08886162541838245, 'op_types': [
                       'Conv2d'], 'op_names': ['features.6.conv.2']},
                   {'sparsity': 0.10687009234802254, 'op_types': [
                       'Conv2d'], 'op_names': ['features.7.conv.0.0']},
                   {'sparsity': 0.11120234857036543, 'op_types': [
                       'Conv2d'], 'op_names': ['features.15.conv.1.0']},
                   {'sparsity': 0.11365381943332029, 'op_types': [
                       'Conv2d'], 'op_names': ['features.16.conv.1.0']},
                   {'sparsity': 0.11715990421744409, 'op_types': [
                       'Conv2d'], 'op_names': ['features.17.conv.1.0']},
                   {'sparsity': 0.11843708882265852, 'op_types': [
                       'Conv2d'], 'op_names': ['features.7.conv.2']},
                   {'sparsity': 0.12239189760322226, 'op_types': [
                       'Conv2d'], 'op_names': ['features.8.conv.0.0']},
                   {'sparsity': 0.13126151468362166, 'op_types': [
                       'Conv2d'], 'op_names': ['features.8.conv.2']},
                   {'sparsity': 0.13896675887295376, 'op_types': [
                       'Conv2d'], 'op_names': ['features.9.conv.0.0']},
                   {'sparsity': 0.140030787684942, 'op_types': [
                       'Conv2d'], 'op_names': ['features.9.conv.2']},
                   {'sparsity': 0.14226554718057852, 'op_types': [
                       'Conv2d'], 'op_names': ['features.10.conv.0.0']},
                   {'sparsity': 0.17579713477725034, 'op_types': [
                       'Conv2d'], 'op_names': ['features.10.conv.2']},
                   {'sparsity': 0.19776658991795798, 'op_types': [
                       'Conv2d'], 'op_names': ['features.11.conv.0.0']},
                   {'sparsity': 0.217601335141064, 'op_types': [
                       'Conv2d'], 'op_names': ['features.11.conv.2']},
                   {'sparsity': 0.21800481760674084, 'op_types': [
                       'Conv2d'], 'op_names': ['features.12.conv.0.0']},
                   {'sparsity': 0.2258469460385748, 'op_types': [
                       'Conv2d'], 'op_names': ['features.12.conv.2']},
                   {'sparsity': 0.22977458912854654, 'op_types': [
                       'Conv2d'], 'op_names': ['features.13.conv.0.0']},
                   {'sparsity': 0.23394197035624212, 'op_types': [
                       'Conv2d'], 'op_names': ['features.13.conv.2']},
                   {'sparsity': 0.23419455191634486, 'op_types': [
                       'Conv2d'], 'op_names': ['features.14.conv.0.0']},
                   {'sparsity': 0.25659318323956287, 'op_types': [
                       'Conv2d'], 'op_names': ['features.14.conv.2']},
                   {'sparsity': 0.3010144701307128, 'op_types': [
                       'Conv2d'], 'op_names': ['features.15.conv.0.0']},
                   {'sparsity': 0.30609157742110915, 'op_types': [
                       'Conv2d'], 'op_names': ['features.15.conv.2']},
                   {'sparsity': 0.3180966817704965, 'op_types': [
                       'Conv2d'], 'op_names': ['features.16.conv.0.0']},
                   {'sparsity': 0.323409539433467, 'op_types': [
                       'Conv2d'], 'op_names': ['features.16.conv.2']},
                   {'sparsity': 0.3581272617655439, 'op_types': ['Conv2d'], 'op_names': ['features.17.conv.0.0']},
                   {'sparsity': 0.3829094816671595, 'op_types': ['Conv2d'], 'op_names': ['features.17.conv.2']},
                   {'sparsity': 0.3970892442274527, 'op_types': ['Conv2d'], 'op_names': ['features.18.0']}]

    pruner = L1FilterPruner(model, config_list)
    # pruner = SimulatedAnnealingPruner(
    #     model, config_list, evaluator=evaluator, pruning_mode=args.pruning_mode, cool_down_rate=0.5, experiment_data_dir=args.experiment_data_dir)
    model_masked = pruner.compress()
    print("Evaluation begins...")
    evaluation_result = evaluator(model_masked)
    print('Evaluation result (masked model): %s' % evaluation_result)
    result['pruned'] = evaluation_result
'''
    if args.fine_tune:
        if args.dataset == 'mnist':
            for epoch in range(args.fine_tune_epochs):
                optimizer = optim.Adadelta(
                    model_masked.parameters(), lr=args.lr)
                train(args, model_masked, device,
                      train_loader, criterion, optimizer, epoch)
                test(model_masked, device, criterion, val_loader)
                scheduler.step()
        elif args.dataset == 'imagenet':
            for epoch in range(args.fine_tune_epochs):
                optimizer = torch.optim.SGD(model_masked.parameters(), lr=0.05,
                                            momentum=0.9,
                                            weight_decay=4e-5)
                train(args, model_masked, device,
                      train_loader, criterion, optimizer, epoch)
                test(model_masked, device, criterion, val_loader)

    evaluation_result = evaluator(model_masked)
    print('Evaluation result (fine tuned): %s' % evaluation_result)
    result['finetuned'] = evaluation_result

    with open(os.path.join(args.experiment_data_dir, 'performance.json'), 'w') as f:
        json.dump(result, f)
'''
