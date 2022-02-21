import random

from torchvision import transforms
from torchvision.datasets import CIFAR10

import searchspace

import nni.retiarii.evaluator.pytorch.lightning as pl
from nni.retiarii.utils import ContextStack
from nni.retiarii.execution.utils import _unpack_if_only_one
from nni.retiarii.mutator import Sampler
from nni.retiarii.nn.pytorch.mutator import extract_mutation_from_pt_module


class RandomSampler(Sampler):
    def __init__(self):
        self.counter = 0

    def choice(self, candidates, *args, **kwargs):
        self.counter += 1
        return random.choice(candidates)


def _test_searchspace_on_cifar10(searchspace):
    model, mutators = extract_mutation_from_pt_module(searchspace)
    for mutator in mutators:
        model = mutator.bind_sampler(RandomSampler()).apply(model)

    mutation = {mut.mutator.label: _unpack_if_only_one(mut.samples) for mut in model.history}
    with ContextStack('fixed', mutation):
        model = model.python_class(**model.python_init_params)

    MEAN = [0.49139968, 0.48215827, 0.44653124]
    STD = [0.24703233, 0.24348505, 0.26158768]
    transf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]

    train_transform = transforms.Compose(transf + normalize)
    valid_transform = transforms.Compose(normalize)

    train_data = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    valid_data = CIFAR10(root="./data", train=False, download=True, transform=valid_transform)

    train_dataloader = pl.DataLoader(train_data, batch_size=16, shuffle=True)
    valid_dataloader = pl.DataLoader(valid_data, batch_size=32)

    evaluator = pl.Classification(
        train_dataloader=train_dataloader,
        val_dataloaders=valid_dataloader,
        max_epochs=1,
        limit_train_batches=0.1
    )
    evaluator.fit(model)


if __name__ == '__main__':
    _test_searchspace_on_cifar10(searchspace.ShuffleNetSpace(num_labels=10, channel_search=True))
