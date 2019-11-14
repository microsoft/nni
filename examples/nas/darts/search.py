from argparse import ArgumentParser

import torch
import torch.nn as nn

import datasets
from nni.nas.pytorch.darts import CnnNetwork, DartsTrainer
from utils import accuracy

if __name__ == "__main__":
    parser = ArgumentParser("darts")
    parser.add_argument("--layers", default=5, type=int)
    parser.add_argument("--nodes", default=4, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--log-frequency", default=1, type=int)
    args = parser.parse_args()

    dataset_train, dataset_valid = datasets.get_dataset("cifar10")

    model = CnnNetwork(3, 16, 10, args.layers, n_nodes=args.nodes)
    criterion = nn.CrossEntropyLoss()

    optim = torch.optim.SGD(model.parameters(), 0.025, momentum=0.9, weight_decay=3.0E-4)
    n_epochs = 50
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, n_epochs, eta_min=0.001)

    trainer = DartsTrainer(model,
                           loss=criterion,
                           metrics=lambda output, target: accuracy(output, target, topk=(1,)),
                           model_optim=optim,
                           lr_scheduler=lr_scheduler,
                           num_epochs=50,
                           dataset_train=dataset_train,
                           dataset_valid=dataset_valid,
                           batch_size=args.batch_size,
                           log_frequency=args.log_frequency)
    trainer.train()
    trainer.export()

# augment step
# ...
