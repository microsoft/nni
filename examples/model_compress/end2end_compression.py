# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
NNI example for combined pruning and quantization to compress a model.
In this example, we show the compression process to first prune a model, then quantize the pruned model.

"""
import argparse
import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from nni.compression.pytorch.utils.counter import count_flops_params
from nni.compression.pytorch import ModelSpeedup

from nni.algorithms.compression.pytorch.pruning import L1FilterPruner
from nni.algorithms.compression.pytorch.quantization import QAT_Quantizer

from models.mnist.naive import NaiveModel
from nni.compression.pytorch.quantization_speedup import ModelSpeedupTensorRT


def get_model_time_cost(model, dummy_input):
    model.eval()
    n_times = 100
    time_list = []
    for _ in range(n_times):
        torch.cuda.synchronize()
        tic = time.time()
        _ = model(dummy_input)
        torch.cuda.synchronize()
        time_list.append(time.time()-tic)
    time_list = time_list[10:]
    return sum(time_list) / len(time_list)


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
            if args.dry_run:
                break


def test(args, model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    acc = 100 * correct / len(test_loader.dataset)

    print('Test Loss: {:.6f}  Accuracy: {}%\n'.format(
        test_loss, acc))
    return acc

def test_trt(engine, test_loader):
    test_loss = 0
    correct = 0
    time_elasped = 0
    for data, target in test_loader:
        output, time = engine.inference(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        time_elasped += time
    test_loss /= len(test_loader.dataset)

    print('Loss: {}  Accuracy: {}%'.format(
        test_loss, 100 * correct / len(test_loader.dataset)))
    print("Inference elapsed_time (whole dataset): {}s".format(time_elasped))

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.experiment_data_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transform),
        batch_size=64,)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transform),
        batch_size=1000)

    # Step1. Model Pretraining
    model = NaiveModel().to(device)
    criterion = torch.nn.NLLLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=args.pretrain_lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    flops, params, _ = count_flops_params(model, (1, 1, 28, 28), verbose=False)

    if args.pretrained_model_dir is None:
        args.pretrained_model_dir = os.path.join(args.experiment_data_dir, f'pretrained.pth')

        best_acc = 0
        for epoch in range(args.pretrain_epochs):
            train(args, model, device, train_loader, criterion, optimizer, epoch)
            scheduler.step()
            acc = test(args, model, device, criterion, test_loader)
            if acc > best_acc:
                best_acc = acc
                state_dict = model.state_dict()

        model.load_state_dict(state_dict)
        torch.save(state_dict, args.pretrained_model_dir)
        print(f'Model saved to {args.pretrained_model_dir}')
    else:
        state_dict = torch.load(args.pretrained_model_dir)
        model.load_state_dict(state_dict)
        best_acc = test(args, model, device, criterion, test_loader)

    dummy_input = torch.randn([1000, 1, 28, 28]).to(device)
    time_cost = get_model_time_cost(model, dummy_input)

    # 125.49 M, 0.85M, 93.29, 1.1012
    print(f'Pretrained model FLOPs {flops/1e6:.2f} M, #Params: {params/1e6:.2f}M, Accuracy: {best_acc: .2f}, Time Cost: {time_cost}')

    # Step2. Model Pruning
    config_list = [{
        'sparsity': args.sparsity,
        'op_types': ['Conv2d']
    }]

    kw_args = {}
    if args.dependency_aware:
        dummy_input = torch.randn([1000, 1, 28, 28]).to(device)
        print('Enable the dependency_aware mode')
        # note that, not all pruners support the dependency_aware mode
        kw_args['dependency_aware'] = True
        kw_args['dummy_input'] = dummy_input

    pruner = L1FilterPruner(model, config_list, **kw_args)
    model = pruner.compress()
    pruner.get_pruned_weights()

    mask_path = os.path.join(args.experiment_data_dir, 'mask.pth')
    model_path = os.path.join(args.experiment_data_dir, 'pruned.pth')
    pruner.export_model(model_path=model_path, mask_path=mask_path)
    pruner._unwrap_model()  # unwrap all modules to normal state

    # Step3. Model Speedup
    m_speedup = ModelSpeedup(model, dummy_input, mask_path, device)
    m_speedup.speedup_model()
    print('model after speedup', model)

    flops, params, _ = count_flops_params(model, dummy_input, verbose=False)
    acc = test(args, model, device, criterion, test_loader)
    time_cost = get_model_time_cost(model, dummy_input)
    print(f'Pruned model FLOPs {flops/1e6:.2f} M, #Params: {params/1e6:.2f}M, Accuracy: {acc: .2f}, Time Cost: {time_cost}')

    # Step4. Model Finetuning
    optimizer = optim.Adadelta(model.parameters(), lr=args.pretrain_lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    best_acc = 0
    for epoch in range(args.finetune_epochs):
        train(args, model, device, train_loader, criterion, optimizer, epoch)
        scheduler.step()
        acc = test(args, model, device, criterion, test_loader)
        if acc > best_acc:
            best_acc = acc
            state_dict = model.state_dict()

    model.load_state_dict(state_dict)
    save_path = os.path.join(args.experiment_data_dir, f'finetuned.pth')
    torch.save(state_dict, save_path)

    flops, params, _ = count_flops_params(model, dummy_input, verbose=True)
    time_cost = get_model_time_cost(model, dummy_input)

    # FLOPs 28.48 M, #Params: 0.18M, Accuracy:  89.03, Time Cost: 1.03
    print(f'Finetuned model FLOPs {flops/1e6:.2f} M, #Params: {params/1e6:.2f}M, Accuracy: {best_acc: .2f}, Time Cost: {time_cost}')
    print(f'Model saved to {save_path}')

    # Step5. Model Quantization via QAT
    config_list = [{
        'quant_types': ['weight', 'output'],
        'quant_bits': {'weight': 8, 'output': 8},
        'op_names': ['conv1']
    }, {
        'quant_types': ['output'],
        'quant_bits': {'output':8},
        'op_names': ['relu1']
    }, {
        'quant_types': ['weight', 'output'],
        'quant_bits': {'weight': 8, 'output': 8},
        'op_names': ['conv2']
    }, {
        'quant_types': ['output'],
        'quant_bits': {'output': 8},
        'op_names': ['relu2']
    }]

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    quantizer = QAT_Quantizer(model, config_list, optimizer)
    quantizer.compress()

    # Step6. Quantization Aware Training
    best_acc = 0
    for epoch in range(1):
        train(args, model, device, train_loader, criterion, optimizer, epoch)
        scheduler.step()
        acc = test(args, model, device, criterion, test_loader)
        if acc > best_acc:
            best_acc = acc
            state_dict = model.state_dict()

    calibration_path = os.path.join(args.experiment_data_dir, 'calibration.pth')
    calibration_config = quantizer.export_model(model_path, calibration_path)
    print("calibration_config: ", calibration_config)

    # Step7. Model Speedup
    batch_size = 32
    input_shape = (batch_size, 1, 28, 28)
    engine = ModelSpeedupTensorRT(model, input_shape, config=calibration_config, batchsize=32)
    engine.compress()

    test_trt(engine, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Example for model comporession')

    # dataset and model
    # parser.add_argument('--dataset', type=str, default='mnist',
    #                     help='dataset to use, mnist, cifar10 or imagenet')
    # parser.add_argument('--data-dir', type=str, default='./data/',
    #                     help='dataset directory')
    parser.add_argument('--pretrained-model-dir', type=str, default=None,
                        help='path to pretrained model')
    parser.add_argument('--pretrain-epochs', type=int, default=10,
                        help='number of epochs to pretrain the model')
    parser.add_argument('--pretrain-lr', type=float, default=1.0,
                        help='learning rate to pretrain the model')

    parser.add_argument('--experiment-data-dir', type=str, default='./experiment_data',
                        help='For saving output checkpoints')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    # parser.add_argument('--multi-gpu', action='store_true', default=False,
    #                     help='run on mulitple gpus')
    # parser.add_argument('--test-only', action='store_true', default=False,
    #                     help='run test only')

    # pruner
    # parser.add_argument('--pruner', type=str, default='l1filter',
    #                     choices=['level', 'l1filter', 'l2filter', 'slim', 'agp',
    #                              'fpgm', 'mean_activation', 'apoz', 'admm'],
    #                     help='pruner to use')
    parser.add_argument('--sparsity', type=float, default=0.5,
                        help='target overall target sparsity')
    parser.add_argument('--dependency-aware', action='store_true', default=False,
                        help='toggle dependency-aware mode')

    # finetuning
    parser.add_argument('--finetune-epochs', type=int, default=5,
                        help='epochs to fine tune')
    # parser.add_argument('--kd', action='store_true', default=False,
    #                     help='quickly check a single pass')
    # parser.add_argument('--kd_T', type=float, default=4,
    #                     help='temperature for KD distillation')
    # parser.add_argument('--finetune-lr', type=float, default=0.5,
    #                     help='learning rate to finetune the model')

    # speedup
    # parser.add_argument('--speedup', action='store_true', default=False,
    #                     help='whether to speedup the pruned model')

    # parser.add_argument('--nni', action='store_true', default=False,
    #                     help="whether to tune the pruners using NNi tuners")

    args = parser.parse_args()
    main(args)
