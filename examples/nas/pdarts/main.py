from argparse import ArgumentParser

import datasets
import torch
import torch.nn as nn
import nni.nas.pytorch as nas
from nni.nas.pytorch.pdarts import PdartsTrainer
from nni.nas.pytorch.darts import CnnNetwork


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = dict()
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res["acc{}".format(k)] = correct_k.mul_(1.0 / batch_size).item()
    return res


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

    optim = torch.optim.SGD(model.parameters(), 0.025,
                            momentum=0.9, weight_decay=3.0E-4)
    n_epochs = 50
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, n_epochs, eta_min=0.001)

    trainer = PdartsTrainer(model,
                            loss=criterion,
                            metrics=lambda output, target: accuracy(
                                output, target, topk=(1,)),
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
