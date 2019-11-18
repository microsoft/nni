from argparse import ArgumentParser

import torch
import torch.nn as nn

import datasets
from macro import GeneralNetwork
from micro import MicroNetwork
from nni.nas.pytorch import enas
from nni.nas.pytorch.callbacks import LearningRateScheduler, ArchitectureCheckpoint
from utils import accuracy, reward_accuracy

if __name__ == "__main__":
    parser = ArgumentParser("enas")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--log-frequency", default=1, type=int)
    parser.add_argument("--search-for", choices=["macro", "micro"], default="macro")
    args = parser.parse_args()

    dataset_train, dataset_valid = datasets.get_dataset("cifar10")
    if args.search_for == "macro":
        model = GeneralNetwork()
        num_epochs = 310
        mutator = None
    elif args.search_for == "micro":
        model = MicroNetwork(num_layers=6, out_channels=20, num_nodes=5, dropout_rate=0.1, use_aux_heads=True)
        num_epochs = 150
        mutator = enas.EnasMutator(model, tanh_constant=1.1, cell_exit_extra_step=True)
    else:
        raise AssertionError

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 0.05, momentum=0.9, weight_decay=1.0E-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.001)

    trainer = enas.EnasTrainer(model,
                               loss=criterion,
                               metrics=accuracy,
                               reward_function=reward_accuracy,
                               optimizer=optimizer,
                               callbacks=[LearningRateScheduler(lr_scheduler), ArchitectureCheckpoint("./checkpoints")],
                               batch_size=args.batch_size,
                               num_epochs=num_epochs,
                               dataset_train=dataset_train,
                               dataset_valid=dataset_valid,
                               log_frequency=args.log_frequency)
    trainer.train_and_validate()
