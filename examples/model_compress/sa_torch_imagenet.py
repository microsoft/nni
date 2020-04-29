from __future__ import print_function

import argparse
import os
import torch
import torch.utils.data
from torchvision import datasets, transforms
import torch.nn.functional as F
import torchvision.models as models

from nni.compression.torch import SimulatedAnnealingPruner

MODEL_DIR = "imagenet_mobilenet.pt"
DATA_DIR = '../data'


def test(model, device, val_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            if batch_idx % args.log_interval == 0:
                print('Validating... {} / {}'.format(batch_idx*len(data), len(val_loader.dataset)))

    test_loss /= len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset), 100. * accuracy))

    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--num_classes', type=int, default=1000, metavar='N',
                        help='number of classes (default 1000)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',  # TODO:14
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--experiment-data-dir', type=str,
                        default='examples/model_compress/experiment_data/', help='For saving experiment data')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {
    }

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    valdir = '/datasets/imagenet/val'
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(pretrained=True).to(device)

    test(model, device, val_loader)

    configure_list = [{
        'sparsity': 0.4,
        'op_types': ['default']  # module types to prune
    }]

    def evaluator(model):
        return test(model=model, device=device, val_loader=val_loader)

    pruner = SimulatedAnnealingPruner(
        model, configure_list, evaluator=evaluator, cool_down_rate=0.9, experiment_data_dir=args.experiment_data_dir)
    pruner.compress()

    pruner.export_model(os.path.join(args.experiment_data_dir, 'model.pth'), os.path.join(args.experiment_data_dir, 'mask.pth'))
    model_pruned = models.mobilenet_v2().to(device)
    model_pruned.load_state_dict(torch.load(os.path.join(args.experiment_data_dir, 'model.pth')))
    evaluation_result = evaluator(model_pruned)
    print('Evaluation result : %s' % evaluation_result)
