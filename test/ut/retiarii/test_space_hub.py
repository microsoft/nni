import random

from torchvision import transforms
from torchvision.datasets import CIFAR10

import nni.retiarii.evaluator.pytorch.lightning as pl
import nni.retiarii.hub.pytorch as searchspace
from nni.retiarii.utils import ContextStack
from nni.retiarii.execution.utils import _unpack_if_only_one
from nni.retiarii.mutator import InvalidMutation, Sampler
from nni.retiarii.nn.pytorch.mutator import extract_mutation_from_pt_module


class RandomSampler(Sampler):
    def __init__(self):
        self.counter = 0

    def choice(self, candidates, *args, **kwargs):
        self.counter += 1
        return random.choice(candidates)


def try_mutation_until_success(base_model, mutators, retry):
    if not retry:
        raise ValueError('Retry exhausted.')
    try:
        model = base_model
        for mutator in mutators:
            model = mutator.bind_sampler(RandomSampler()).apply(model)
        return model
    except InvalidMutation:
        return try_mutation_until_success(base_model, mutators, retry - 1)


def _test_searchspace_on_cifar10(searchspace, resize_to_imagenet=False):
    model, mutators = extract_mutation_from_pt_module(searchspace)
    model = try_mutation_until_success(model, mutators, 10)

    mutation = {mut.mutator.label: _unpack_if_only_one(mut.samples) for mut in model.history}
    print('Selected model:', mutation)
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

    if resize_to_imagenet:
        normalize = [
            transforms.Resize((224, 224)),
        ] + normalize

    train_transform = transforms.Compose(transf + normalize)
    valid_transform = transforms.Compose(normalize)

    train_data = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    valid_data = CIFAR10(root="./data", train=False, download=True, transform=valid_transform)

    train_dataloader = pl.DataLoader(train_data, batch_size=4, shuffle=True)
    valid_dataloader = pl.DataLoader(valid_data, batch_size=6)

    evaluator = pl.Classification(
        train_dataloader=train_dataloader,
        val_dataloaders=valid_dataloader,
        export_onnx=False,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=3,
    )
    evaluator.fit(model)


if __name__ == '__main__':
    ss = searchspace.MobileNetV3Space()
    _test_searchspace_on_cifar10(ss, resize_to_imagenet=True)
