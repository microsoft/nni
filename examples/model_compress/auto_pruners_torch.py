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
from nni.compression.torch import L1FilterPruner, SimulatedAnnealingPruner, ADMMPruner, NetAdaptPruner, AutoCompressPruner
from nni.compression.torch import ModelSpeedup


def get_data(args):
    '''
    get data
    '''
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

    return train_loader, val_loader, criterion


def train(args, model, device, train_loader, criterion, optimizer, epoch, callback=None):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if callback:
            callback()
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


def get_trained_model(args, device, train_loader, val_loader, criterion):
    if args.model == 'LeNet':
        model = LeNet().to(device)
        optimizer = torch.optim.Adadelta(model.parameters(), lr=1)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
        for epoch in range(args.pretrain_epochs):
            train(args, model, device, train_loader,
                  criterion, optimizer, epoch)
            scheduler.step()
    elif args.model == 'vgg16':
        model = VGG(depth=16).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                                    momentum=0.9,
                                    weight_decay=5e-4)
        scheduler = MultiStepLR(
            optimizer, milestones=[int(args.pretrain_epochs*0.5), int(args.pretrain_epochs*0.75)], gamma=0.1)
        for epoch in range(args.pretrain_epochs):
            train(args, model, device, train_loader,
                  criterion, optimizer, epoch)
            scheduler.step()
    elif args.model == 'resnet18':
        model = models.resnet18(pretrained=False, num_classes=10).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                                    momentum=0.9,
                                    weight_decay=5e-4)
        scheduler = MultiStepLR(
            optimizer, milestones=[int(args.pretrain_epochs*0.5), int(args.pretrain_epochs*0.75)], gamma=0.1)
        for epoch in range(args.pretrain_epochs):
            train(args, model, device, train_loader,
                  criterion, optimizer, epoch)
            scheduler.step()
    elif args.model == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True).to(device)

    if args.save_model:
        torch.save(model.state_dict(), os.path.join(
            args.experiment_data_dir, 'model_trained.pth'))
        print('Model trained saved to %s', args.experiment_data_dir)

    return model, optimizer


def get_dummy_input(args, device):
    if args.model == 'LeNet':
        dummy_input = torch.randn(
            [args.test_batch_size, 1, 28, 28]).to(device)
    elif args.model == 'vgg16':
        dummy_input = torch.randn(
            [args.test_batch_size, 3, 32, 32]).to(device)
    elif args.model == 'resnet18':
        dummy_input = torch.randn(
            [args.test_batch_size, 3, 32, 32]).to(device)
    elif args.model == 'mobilenet_v2':
        dummy_input = torch.randn(
            [args.test_batch_size, 3, 32, 32]).to(device)

    return dummy_input


def main(args):
    # prepare dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, criterion = get_data(args)

    torch.manual_seed(0)

    model, optimizer = get_trained_model(
        args, device, train_loader, val_loader, criterion)

    def fine_tuner(model, epochs):
        for epoch in range(epochs):
            train(args, model, device, train_loader,
                  criterion, optimizer, epoch)

    def short_term_fine_tuner(model):
        return fine_tuner(model, epochs=1)

    def trainer(model, optimizer, criterion, epoch, callback):
        return train(args, model, device, train_loader, criterion, optimizer, epoch=epoch, callback=callback)

    def evaluator(model):
        return test(model, device, criterion, val_loader)

    result = {}

    evaluation_result = evaluator(model)
    print('Evaluation result (original model): %s' % evaluation_result)
    result['original'] = evaluation_result

    # module types to prune, only "Conv2d" supported for channel pruning
    if args.pruning_mode == 'channel':
        op_types = ['Conv2d']
    elif args.pruning_mode == 'fine_grained':
        op_types = ['default']

    config_list = [{
        'sparsity': args.sparsity,
        'op_types': op_types
    }]
    dummy_input = get_dummy_input(args, device)

    if args.pruner == 'L1FilterPruner':
        pruner = L1FilterPruner(
            model, config_list)
    elif args.pruner == 'NetAdaptPruner':
        pruner = NetAdaptPruner(model, config_list, fine_tuner=short_term_fine_tuner, evaluator=evaluator,
                                pruning_mode=args.pruning_mode, experiment_data_dir=args.experiment_data_dir)
    elif args.pruner == 'ADMMPruner':
        # users are free to change the config here
        if args.model == 'LeNet':
            if args.pruning_mode == 'channel':
                config_list = [{
                    'sparsity': 0.8,
                    'op_types': ['Conv2d'],
                    'op_names': ['conv1']
                }, {
                    'sparsity': 0.92,
                    'op_types': ['Conv2d'],
                    'op_names': ['conv2']
                }]
            elif args.pruning_mode == 'fine_grained':
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
        else:
            raise ValueError('Example only implemented for LeNet.')
        pruner = ADMMPruner(
            model, config_list, trainer=trainer, optimize_iterations=2, training_epochs=2)
    elif args.pruner == 'SimulatedAnnealingPruner':
        pruner = SimulatedAnnealingPruner(
            model, config_list, evaluator=evaluator, pruning_mode=args.pruning_mode,
            cool_down_rate=args.cool_down_rate, experiment_data_dir=args.experiment_data_dir)
    elif args.pruner == 'AutoCompressPruner':
        pruner = AutoCompressPruner(
            model, config_list, trainer=trainer, evaluator=evaluator, dummy_input=dummy_input,
            optimize_iterations=3, optimize_mode='maximize', pruning_mode=args.pruning_mode,
            cool_down_rate=args.cool_down_rate, admm_optimize_iterations=30, admm_training_epochs=5,
            experiment_data_dir=args.experiment_data_dir)
    else:
        raise ValueError(
            "Please use L1FilterPruner, NetAdaptPruner, SimulatedAnnealingPruner, ADMMPruner or AutoCompressPruner in this example.")

    # pruner.compress() returns the masked model
    # but for AutoCompressPruner, pruner.compress() returns directly the pruned model
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
    if args.speed_up and args.pruner != 'AutoCompressPruner':
        if args.model == 'LeNet':
            model = LeNet().to(device)
        elif args.model == 'vgg16':
            model = VGG(depth=16).to(device)
        elif args.model == 'resnet18':
            model = models.resnet18(
                pretrained=False, num_classes=10).to(device)
        elif args.model == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=True).to(device)

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
                        help='pruner to use, L1FilterPruner, NetAdaptPruner, SimulatedAnnealingPruner, ADMMPruner or AutoCompressPruner')
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

    parser.add_argument('--dataset', type=str, default='mnist',
                        help='dataset to use, mnist, cifar10 or imagenet (default MNIST)')
    parser.add_argument('--model', type=str, default='LeNet',
                        help='model to use, LeNet, vgg16, resnet18 or mobilenet_v2')
    parser.add_argument('--fine-tune', type=str2bool, default=True,
                        help='whether to fine-tune the pruned model')
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
    parser.add_argument('--pretrain-epochs', type=int, default=1,
                        help='number of epochs to pretrain the model')
    parser.add_argument('--log-interval', type=int, default=200,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', type=str2bool, default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()

    if not os.path.exists(args.experiment_data_dir):
        os.makedirs(args.experiment_data_dir)

    main(args)
