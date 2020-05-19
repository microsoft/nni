from __future__ import print_function

import argparse
import os
import torch
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from models.mnist.lenet import LeNet
from nni.compression.torch import LevelPruner, L1FilterPruner

MODEL_DIR = "mnist_lenet.pt"
DATA_DIR = '../data'


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, val_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
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
    parser.add_argument('--experiment-data-dir', type=str,
                        default='examples/model_compress/experiment_data/', help='For saving experiment data')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {
    }

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, val_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), MODEL_DIR)

    configure_list = [{
        'sparsity': 0.5,
        'op_types': ['Conv2d']  # module types to prune
    }]

    def evaluator(model):
        return test(model=model, device=device, val_loader=val_loader)

    pruner = L1FilterPruner(model, configure_list)
    model_pruned = pruner.compress()

    evaluation_result = evaluator(model_pruned)
    print('Evaluation result (bound_model): %s' % evaluation_result)

    # load & save
    pruner.export_model(os.path.join(args.experiment_data_dir, 'model.pth'),
                        os.path.join(args.experiment_data_dir, 'mask.pth'))
    model_pruned = LeNet().to(device)
    model_pruned.load_state_dict(torch.load(
        os.path.join(args.experiment_data_dir, 'model.pth')))
    evaluation_result = evaluator(model_pruned)
    print('Evaluation result (exported_model): %s' % evaluation_result)

    # model speed up
    from nni.compression.speedup.torch import ModelSpeedup
    from nni.compression.torch import apply_compression_results

    model = LeNet().to(device)
    # model.load_state_dict(torch.load(os.path.join(args.experiment_data_dir, 'model.pth')))

    dummy_input = torch.randn([args.test_batch_size, 1, 28, 28]).to(device)
    masks_file = os.path.join(args.experiment_data_dir, 'mask.pth')

    # apply_compression_results(model, masks_file, device)
    m_speedup = ModelSpeedup(model, dummy_input, masks_file, device)
    m_speedup.speedup_model()
    evaluation_result = evaluator(m_speedup)
    print('Evaluation result (speed up model): %s' % evaluation_result)
