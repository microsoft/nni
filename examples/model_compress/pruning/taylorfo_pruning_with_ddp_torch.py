# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

'''
NNI example for supported DistributedDataParallel pruning.
In this example, we use TaylorFo pruner to show the end-to-end ddp pruning process: pre-training -> pruning -> fine-tuning.
Note that pruners use masks to simulate the real pruning. In order to obtain a real compressed model, model speedup is required.

'''
import argparse
import time
import functools
from typing import Callable
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import MultiStepLR

import nni
from nni.compression.pytorch import ModelSpeedup
from nni.compression.pytorch.utils import count_flops_params
from nni.compression.pytorch.pruning import TaylorFOWeightPruner
from nni.compression.pytorch.utils import TorchEvaluator


#############  Create dataloaders, optimizer, training and evaluation function ############

class Mnist(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.fc1 = torch.nn.Linear(4 * 4 * 50, 500)
        self.fc2 = torch.nn.Linear(500, 10)
        self.relu1 = torch.nn.ReLU6()
        self.relu2 = torch.nn.ReLU6()
        self.relu3 = torch.nn.ReLU6()
        self.max_pool1 = torch.nn.MaxPool2d(2, 2)
        self.max_pool2 = torch.nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.max_pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.max_pool2(x)
        x = x.view(x.shape[0], -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


def create_dataloaders():
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # training dataloader
    training_dataset = datasets.MNIST('data', train=True, download=True, transform=trans)
    training_sampler = torch.utils.data.distributed.DistributedSampler(training_dataset)
    training_dataloader = torch.utils.data.DataLoader(training_dataset, \
        batch_size=64, sampler=training_sampler)
    # validation dataloader
    validation_dataset = datasets.MNIST('data', train=False, transform=trans)
    validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, \
        batch_size=1000, sampler=validation_sampler)

    return training_dataloader, validation_dataloader

def training(
    training_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    max_steps: int = None, max_epochs: int = None,
    local_rank: int = -1,
    save_best_model: bool = False, save_path: str = None,
    log_path: str = None,
    evaluation_func=None,
):
    model.train()
    current_step = 0
    best_acc = 0.

    for current_epoch in range(max_epochs if max_epochs else 2):
        for (data, target) in training_dataloader:
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            if lr_scheduler:
                lr_scheduler.step()

            current_step += 1

        # evaluation for every 1000 steps
        if current_step % 1000 == 0 or current_step % len(training_dataloader) == 0:
            acc = evaluation_func(validation_dataloader, model)
            with log_path.open('a+') as f:
                msg = '[{}] Epoch {}, Step {}: Acc: {} Loss:{}\n'.format(time.asctime(time.localtime(time.time())), \
                    current_epoch, current_step, acc, loss.item())
                f.write(msg)
            if save_best_model and best_acc < acc:
                assert save_path is not None
                if local_rank == 0:
                    torch.save(model.module.state_dict(), save_path)
                best_acc = acc

        if max_steps and current_step >= max_steps:
            return best_acc

    return best_acc

def evaluation(validation_dataloader: DataLoader, model: nn.Module):
    training = model.training
    model.eval()

    correct = 0.0
    with torch.no_grad():
        for data, target in validation_dataloader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    acc = 100 * correct / len(validation_dataloader.dataset)
    # average acc in different local_ranks
    average_acc = torch.tensor([acc]).cuda()
    dist.all_reduce(average_acc, op=dist.ReduceOp.SUM)
    world_size = dist.get_world_size()
    average_acc = average_acc / world_size

    print('Average Accuracy: {}%\n'.format(average_acc.item()))
    model.train(training)

    return average_acc.item()

def optimizer_scheduler_generator(model, _lr=0.1, _momentum=0.9, _weight_decay=5e-4, total_epoch=160):
    optimizer = torch.optim.SGD(model.parameters(), lr=_lr, momentum=_momentum, weight_decay=_weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[int(total_epoch * 0.5), int(total_epoch * 0.75)], gamma=0.1)
    return optimizer, scheduler

def retrain_model(
    args,
    local_rank: int,
    model: nn.Module = None,
):
    # create an ddp model
    if model is None:  # pretraining process
        model = Mnist().cuda()
        log_save_path = "pretraining.log"
        model_save_path = "pretraining_best_model.pth"
        epochs = args.pretrain_epochs
        lr = args.pretraining_lr
    else: # finetune process
        log_save_path = "finetune.log"
        model_save_path = "finetune_best_model.pth"
        epochs = args.finetune_epochs
        lr = args.finetune_lr

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # create dataloaders
    training_dataloader, validation_dataloader = create_dataloaders()
    # create optimizer, lr_scheduler and criterion
    optimizer, lr_scheduler = optimizer_scheduler_generator(model, \
        _lr=lr, total_epoch=epochs)
    criterion = torch.nn.CrossEntropyLoss()

    # training and evaluation process
    best_acc = training(training_dataloader, validation_dataloader, model, optimizer, criterion, lr_scheduler,\
        args.max_steps, epochs, local_rank, save_best_model=True, \
        save_path=Path(args.log_dir) / model_save_path, \
        log_path=Path(args.log_dir) / log_save_path, \
        evaluation_func=evaluation)

    # compute params and FLOPs
    flops, params, _ = count_flops_params(model, torch.randn([32, 1, 28, 28]).cuda())

    return flops, params, best_acc

def pruned_model_process(args, local_rank):
    # load the pretrained model
    model = Mnist().cuda()
    state_dict = torch.load(Path(args.log_dir) / "pretraining_best_model.pth")
    model.load_state_dict(state_dict)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    # create dataloaders
    training_dataloader, validation_dataloader = create_dataloaders()
    # build a config_list
    config_list = [{'total_sparsity': 0.7, 'op_types': ['Conv2d']}]
    # create an evaluator
    taylor_training = functools.partial(
        training,
        training_dataloader,
        validation_dataloader,
        local_rank = local_rank,
        log_path = Path(args.log_dir) / "taylor_pruning.log",
        evaluation_func = evaluation,
    )
    traced_optimizer = nni.trace(torch.optim.SGD)(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    evaluator = TorchEvaluator(taylor_training, traced_optimizer, criterion)
    # create an taylor pruner
    pruner = TaylorFOWeightPruner(model=model, config_list=config_list,
                evaluator=evaluator, training_steps=args.pruner_training_steps)
    _, masks = pruner.compress()
    pruner.show_pruned_weights()
    pruner._unwrap_model()
    #speedup
    sub_module = ModelSpeedup(model, dummy_input=torch.rand([32, 1, 28, 28]).cuda(), masks_file=masks).speedup_model()

    return sub_module

def main():
    parser = argparse.ArgumentParser(description='PyTorch Example for model comporession with ddp')
    parser.add_argument('--finetune_lr', type=float, default=0.01,
                        help='the learning rate in the fine-tune process')
    parser.add_argument('--pretraining_lr', type=float, default=0.01,
                        help='the learning rate in the pretraining process')
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='the parameter in the Adam optimizer')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='the max number of training steps')
    parser.add_argument('--log_dir', type=str, default='./mnist_infos',
                        help='the base path for saving files')
    parser.add_argument('--pruner_training_steps', type=int, default=1000,
                        help='the number of training steps in the pruning process')
    parser.add_argument('--pretrain_epochs', type=int, default=5,
                        help='number of epochs to pretrain the model')
    parser.add_argument('--finetune_epochs', type=int, default=20,
                        help='number of epochs to fine-tune the model')
    args = parser.parse_args()

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    #init ddp
    dist.init_process_group(backend='nccl')
    # get local_rank
    rank = dist.get_rank()
    local_rank = rank % torch.cuda.device_count()
    print(f"local_rank:{local_rank}")
    torch.cuda.set_device(local_rank)

    print('\n' + '=' * 50 + ' START TO TRAIN THE MODEL ' + '=' * 50)
    original_flops, original_params, original_best_acc = retrain_model(args, local_rank)

    # # Start to prune and speedup
    print('\n' + '=' * 50 + ' START TO PRUNE THE BEST ACCURACY PRETRAINED MODEL ' + '=' * 50)
    model = pruned_model_process(args, local_rank)
    print('\n' + '=' * 50 + ' START TO FINE TUNE THE MODEL ' + '=' * 50)
    finetuned_flops, finetuned_params, finetuned_best_acc = retrain_model(args, local_rank, model.cuda())
    print(f'Pretrained model FLOPs {original_flops/1e6:.2f} M, #Params: {original_params/1e6:.2f}M, Accuracy: {original_best_acc: .2f}%')
    print(f'Finetuned model FLOPs {finetuned_flops/1e6:.2f} M, #Params: {finetuned_params/1e6:.2f}M, Accuracy: {finetuned_best_acc: .2f}%')



if __name__ == '__main__':
    main()
