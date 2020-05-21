# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
'''
Examples for automatic pruners
'''

import argparse
import os
import json
import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torchvision import datasets, transforms, models

from models.mnist.lenet import LeNet
from models.cifar10.vgg import VGG
from nni.compression.torch import L1FilterPruner, SimulatedAnnealingPruner, NetAdaptPruner, ADMMPruner
from nni.compression.speedup.torch import ModelSpeedup


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
            test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset), 100. * accuracy))

    return accuracy


def main(args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {
    }

    # prepare dataset
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
        criterion = torch.nn.NLLLoss()
    elif args.dataset == 'cifar10':
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data_dir, train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data_dir, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False, **kwargs)
        criterion = torch.nn.CrossEntropyLoss()
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
        criterion = torch.nn.CrossEntropyLoss()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == 'LeNet':
        model = LeNet().to(device)
        optimizer = torch.optim.Adadelta(model.parameters(), lr=1)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
        for epoch in range(args.epochs):
            train(args, model, device, train_loader,
                  criterion, optimizer, epoch)
            scheduler.step()
        torch.save(model.state_dict(), os.path.join(
            args.experiment_data_dir, 'model_trained.pth'))
    elif args.dataset == 'vgg16':
        model = models.vgg16(pretrained=False, num_classes=10).to(device)
        # model = VGG(depth=16).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                                    momentum=0.9,
                                    weight_decay=5e-4)
        scheduler = MultiStepLR(
            optimizer, milestones=[int(args.epochs*0.5), int(args.epochs*0.75)], gamma=0.1)
        for epoch in range(args.epochs):
            train(args, model, device, train_loader,
                  criterion, optimizer, epoch)
            scheduler.step()
        if args.save_model:
            torch.save(model.state_dict(), os.path.join(
                args.experiment_data_dir, 'model_trained.pth'))
            print('Model trained saved to %s', args.experiment_data_dir)
    elif args.model == 'resnet18':
        model = models.resnet18(pretrained=False, num_classes=10).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                                    momentum=0.9,
                                    weight_decay=5e-4)
        scheduler = MultiStepLR(
            optimizer, milestones=[int(args.epochs*0.5), int(args.epochs*0.75)], gamma=0.1)
        for epoch in range(args.epochs):
            train(args, model, device, train_loader,
                  criterion, optimizer, epoch)
            scheduler.step()
        if args.save_model:
            torch.save(model.state_dict(), os.path.join(
                args.experiment_data_dir, 'model_trained.pth'))
            print('Model trained saved to %s', args.experiment_data_dir)
    elif args.dataset == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True).to(device)

    def fine_tuner(model, epochs):
        if args.dataset == 'mnist':
            optimizer = torch.optim.Adadelta(model.parameters(), lr=1)
        elif args.dataset == 'cifar10':
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                                        momentum=0.9,
                                        weight_decay=5e-4)
        elif args.dataset == 'imagenet':
            # TODO: decay lr
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                                        momentum=0.9,
                                        weight_decay=5e-4)
        for epoch in range(epochs):
            train(args, model, device, train_loader,
                  criterion, optimizer, epoch)

    def short_term_fine_tuner(model):
        return fine_tuner(model, epochs=1)

    def evaluator(model):
        return test(model, device, criterion, val_loader)

    result = {}

    evaluation_result = evaluator(model)
    print('Evaluation result (original model): %s' % evaluation_result)
    result['original'] = evaluation_result

    # module types to prune, only "Conv2d" supported for channel pruning
    if args.pruner == 'ADMMPruner':
        config_list = [{
            'sparsity': 0.8,
            'op_names': ['conv1']
        }, {
            'sparsity': 0.92,
            'op_names': ['conv2']
        }, {
            'sparsity': 0.991,
            'op_names': ['fc1']
        }, {
            'sparsity': 0.93,
            'op_names': ['fc2']
        }]

    if args.pruner == 'L1FilterPruner':
        pruner = L1FilterPruner(
            model, config_list)
    elif args.pruner == 'ADMMPruner':
        pruner = ADMMPruner(
            model, config_list, experiment_data_dir=args.experiment_data_dir)

    model_masked = pruner.compress()
    evaluation_result = evaluator(model_masked)
    print('Evaluation result (masked model): %s' % evaluation_result)
    result['pruned'] = evaluation_result

    if args.save_model:
        pruner.export_model(os.path.join(args.experiment_data_dir, 'model_masked.pth'), os.path.join(
            args.experiment_data_dir, 'mask.pth'))
        print('Masked model saved to %s', args.experiment_data_dir)

    if args.fine_tune:
        if args.dataset == 'mnist':
            optimizer = torch.optim.Adadelta(
                model_masked.parameters(), lr=1)
            scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
            for epoch in range(args.fine_tune_epochs):
                train(args, model_masked, device,
                      train_loader, criterion, optimizer, epoch)
                test(model_masked, device, criterion, val_loader)
                scheduler.step()
        elif args.dataset == 'cifar10':
            optimizer = torch.optim.SGD(model_masked.parameters(), lr=0.01,
                                        momentum=0.9,
                                        weight_decay=5e-4)
            scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
            for epoch in range(args.fine_tune_epochs):
                train(args, model_masked, device,
                      train_loader, criterion, optimizer, epoch)
                scheduler.step()
                test(model_masked, device, criterion, val_loader)
        elif args.dataset == 'imagenet':
            for epoch in range(args.fine_tune_epochs):
                optimizer = torch.optim.SGD(model_masked.parameters(), lr=0.05,
                                            momentum=0.9,
                                            weight_decay=5e-4)
                train(args, model_masked, device,
                      train_loader, criterion, optimizer, epoch)
                test(model_masked, device, criterion, val_loader)

    evaluation_result = evaluator(model_masked)
    print('Evaluation result (fine tuned): %s' % evaluation_result)
    result['finetuned'] = evaluation_result

    if args.save_model:
        pruner.export_model(os.path.join(
            args.experiment_data_dir, 'model_fine_tuned.pth'), os.path.join(args.experiment_data_dir, 'mask.pth'))
        print('Fined tuned model saved to %s', args.experiment_data_dir)

    # model speed up
    print("args.speed_up: %s", args.speed_up)
    if args.speed_up:
        if args.model == 'LeNet':
            model = LeNet().to(device)
            dummy_input = torch.randn(
                [args.test_batch_size, 1, 28, 28]).to(device)
        elif args.model == 'vgg16':
            model = models.vgg16(
                pretrained=False, num_classes=10).to(device)
            # model = VGG(depth=16).to(device)
            dummy_input = torch.randn(
                [args.test_batch_size, 3, 32, 32]).to(device)
        elif args.model == 'resnet18':
            model = models.resnet18(
                pretrained=False, num_classes=10).to(device)
            dummy_input = torch.randn(
                [args.test_batch_size, 3, 32, 32]).to(device)
        elif args.model == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=True).to(device)
            dummy_input = torch.randn(
                [args.test_batch_size, 3, 32, 32]).to(device)

        model.load_state_dict(torch.load(os.path.join(
            args.experiment_data_dir, 'model_fine_tuned.pth')))

        masks_file = os.path.join(args.experiment_data_dir, 'mask.pth')

        m_speedup = ModelSpeedup(model, dummy_input, masks_file, device)
        m_speedup.speedup_model()
        evaluation_result = evaluator(model)
        print('Evaluation result (speed up model): %s' % evaluation_result)
        result['speedup'] = evaluation_result

        torch.save(model.state_dict(), os.path.join(
            args.experiment_data_dir, 'model_speed_up.pth'))
        print('Speed up model saved to %s', args.experiment_data_dir)

    with open(os.path.join(args.experiment_data_dir, 'performance.json'), 'w+') as f:
        json.dump(result, f)


if __name__ == '__main__':
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(
        description='PyTorch Example for SimulatedAnnealingPruner')

    parser.add_argument('--pruner', type=str, default='SimulatedAnnealingPruner',
                        help='pruner to use, L1FilterPruner, SimulatedAnnealingPruner or NetAdaptPruner')
    parser.add_argument('--pruning-mode', type=str, default='channel',
                        help='pruning mode, channel or fine_grained')
    parser.add_argument('--sparsity', type=float, default=0.3,
                        help='overall target sparsity')
    parser.add_argument('--speed-up', type=str2bool, default=True,
                        help='Whether to speed-up the pruned model')

    # param for SimulatedAnnealingPruner
    parser.add_argument('--cool-down-rate', type=float, default=0.9,
                        help='cool down rate')
    # param for NetAdaptPruner
    parser.add_argument('--pruning-step', type=float, default=0.05,
                        help='pruning_step of NetAdaptPruner')

    # LeNet, VGG16 and MobileNetV2 used for these three different datasets respectively
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='dataset to use, mnist, cifar10 or imagenet (default MNIST)')
    parser.add_argument('--model', type=str, default='LeNet',
                        help='model to use, LeNet, vgg16, resnet18 or mobilenet_v2')
    parser.add_argument('--fine-tune', type=str2bool, default=True,
                        help='Whether to fine-tune the pruned model')
    parser.add_argument('--fine-tune-epochs', type=int, default=10,
                        help='epochs to fine tune')
    parser.add_argument('--data-dir', type=str,
                        default='/datasets/')
    parser.add_argument('--experiment-data-dir', type=str,
                        default='./', help='For saving experiment data')

    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64,
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--log-interval', type=int, default=200,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', type=str2bool, default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()

    if not os.path.exists(args.experiment_data_dir):
        os.makedirs(args.experiment_data_dir)

    main(args)
