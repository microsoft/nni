import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from models.cifar10.vgg import VGG
from nni.compression.speedup.torch import ModelSpeedup
from nni.compression.torch import apply_compression_results

torch.manual_seed(0)
use_mask = False

def l1filter_speedup(masks_file, model_checkpoint):
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
        batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=200, shuffle=False)

    model = VGG(depth=16)
    model.to(device)
    model.eval()

    dummy_input = torch.randn(64, 3, 32, 32)
    if use_mask:
        apply_compression_results(model, masks_file)
        dummy_input = dummy_input.to(device)
        start = time.time()
        for _ in range(1):
            out = model(dummy_input)
        print(out.size(), out)
        #print('mask elapsed time: ', time.time() - start)
        return
    else:
        #print("model before: ", model)
        m_speedup = ModelSpeedup(model, dummy_input.to(device), masks_file)
        m_speedup.speedup_model()
        #print("model after: ", model)
        dummy_input = dummy_input.to(device)
        start = time.time()
        for _ in range(1):
            out = model(dummy_input)
        print(out.size(), out)
        #print('speedup elapsed time: ', time.time() - start)
        return

def fpgm_speedup(masks_file, model_checkpoint):
    from fpgm_torch_mnist import Mnist
    device = torch.device('cpu')
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=trans),
        batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=trans),
        batch_size=1000, shuffle=True)

    model = Mnist()
    model.to(device)
    model.print_conv_filter_sparsity()

    dummy_input = torch.randn(64, 1, 28, 28)
    if use_mask:
        apply_compression_results(model, masks_file)
        dummy_input = dummy_input.to(device)
        start = time.time()
        for _ in range(40):
            out = model(dummy_input)
        print('mask elapsed time: ', time.time() - start)
        #print(out.size(), out)
        return
    else:
        m_speedup = ModelSpeedup(model, dummy_input.to(device), masks_file)
        m_speedup.speedup_model()
        dummy_input = dummy_input.to(device)
        start = time.time()
        for _ in range(40):
            out = model(dummy_input)
        print('speedup elapsed time: ', time.time() - start)
        #print(out.size(), out)
        return

def slim_speedup(masks_file, model_checkpoint):
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
        batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=200, shuffle=False)

    model = VGG(depth=19)
    model.to(device)
    model.eval()

    dummy_input = torch.randn(64, 3, 32, 32)
    if use_mask:
        apply_compression_results(model, masks_file)
        dummy_input = dummy_input.to(device)
        start = time.time()
        for _ in range(32):
            out = model(dummy_input)
        #print(out.size(), out)
        print('mask elapsed time: ', time.time() - start)
        return
    else:
        #print("model before: ", model)
        m_speedup = ModelSpeedup(model, dummy_input.to(device), masks_file)
        m_speedup.speedup_model()
        #print("model after: ", model)
        dummy_input = dummy_input.to(device)
        start = time.time()
        for _ in range(32):
            out = model(dummy_input)
        #print(out.size(), out)
        print('speedup elapsed time: ', time.time() - start)
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser("speedup")
    parser.add_argument("--example_name", type=str, default="l1filter", help="the name of pruning example")
    parser.add_argument("--masks_file", type=str, default=None, help="the path of the masks file")
    parser.add_argument("--model_checkpoint", type=str, default=None, help="the path of checkpointed model")
    args = parser.parse_args()
    
    if args.example_name == 'slim':
        if args.masks_file is None:
            args.masks_file = 'mask_vgg19_cifar10.pth'
        slim_speedup(args.masks_file, args.model_checkpoint)
    elif args.example_name == 'fpgm':
        if args.masks_file is None:
            args.masks_file = 'mask.pth'
        fpgm_speedup(args.masks_file, args.model_checkpoint)
    elif args.example_name == 'l1filter':
        if args.masks_file is None:
            args.masks_file = 'mask_vgg16_cifar10.pth'
        l1filter_speedup(args.masks_file, args.model_checkpoint)
    else:
        raise ValueError('unsupported example_name: {}'.format(args.example_name))
