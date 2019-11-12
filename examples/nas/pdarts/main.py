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
    parser.add_argument('--add_layers', action='append',
                        default=[0, 6, 12], help='add layers')
    parser.add_argument("--nodes", default=4, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--log-frequency", default=1, type=int)
    args = parser.parse_args()

    dataset_train, dataset_valid = datasets.get_dataset("cifar10")


    def model_creator(layers, n_nodes):
        model = CnnNetwork(3, 16, 10, layers, n_nodes=n_nodes)
        loss = nn.CrossEntropyLoss()

        model_optim = torch.optim.SGD(model.parameters(), 0.025,
                                      momentum=0.9, weight_decay=3.0E-4)
        n_epochs = 50
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            model_optim, n_epochs, eta_min=0.001)
        return model, loss, model_optim, lr_scheduler

    trainer = PdartsTrainer(model_creator,
                            metrics=lambda output, target: accuracy(
                                output, target, topk=(1,)),
                            num_epochs=50,
                            pdarts_num_layers=[0, 6, 12],
                            pdarts_num_to_drop=[3, 2, 2],
                            dataset_train=dataset_train,
                            dataset_valid=dataset_valid,
                            layers=args.layers,
                            n_nodes=args.nodes,
                            batch_size=args.batch_size,
                            log_frequency=args.log_frequency)
    trainer.train()
    trainer.export()

# augment step
# ...
