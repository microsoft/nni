import random

from torchvision import transforms
from torchvision.datasets import CIFAR10, FakeData

import nni
import nni.runtime.platform.test
import nni.retiarii.evaluator.pytorch.lightning as pl
import nni.retiarii.hub.pytorch as searchspace
from nni.retiarii.utils import ContextStack
from nni.retiarii.execution.utils import _unpack_if_only_one
from nni.retiarii.mutator import InvalidMutation, Sampler
from nni.retiarii.nn.pytorch.mutator import extract_mutation_from_pt_module


def _reset():
    # this is to not affect other tests in sdk
    nni.trial._intermediate_seq = 0
    nni.trial._params = {'foo': 'bar', 'parameter_id': 0}
    nni.runtime.platform.test._last_metric = None


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


def _test_searchspace_on_dataset(searchspace, dataset='cifar10'):
    _reset()

    # dataset supports cifar10 and fake-imagenet
    model, mutators = extract_mutation_from_pt_module(searchspace)
    model = try_mutation_until_success(model, mutators, 10)

    mutation = {mut.mutator.label: _unpack_if_only_one(mut.samples) for mut in model.history}
    print('Selected model:', mutation)
    with ContextStack('fixed', mutation):
        model = model.python_class(**model.python_init_params)

    if dataset == 'cifar10':
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

    elif dataset == 'fake-imagenet':
        train_data = FakeData(size=200, image_size=(3, 224, 224), num_classes=1000, transform=transforms.ToTensor())
        valid_data = FakeData(size=200, image_size=(3, 224, 224), num_classes=1000, transform=transforms.ToTensor())

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


def test_nasbench101():
    ss = searchspace.NasBench101()
    _test_searchspace_on_dataset(ss)


def test_nasbench201():
    ss = searchspace.NasBench101()
    _test_searchspace_on_dataset(ss)


def test_nasnet():
    _test_searchspace_on_dataset(searchspace.NASNet())
    _test_searchspace_on_dataset(searchspace.ENAS())
    _test_searchspace_on_dataset(searchspace.AmoebaNet())
    _test_searchspace_on_dataset(searchspace.PNAS())
    _test_searchspace_on_dataset(searchspace.DARTS())


def test_nasnet_fixwd():
    # minimum
    ss = searchspace.DARTS(width=16, num_cells=4)
    _test_searchspace_on_dataset(ss)

    # medium
    ss = searchspace.DARTS(width=16, num_cells=12)
    _test_searchspace_on_dataset(ss)


def test_nasnet_imagenet():
    ss = searchspace.ENAS(dataset='imagenet')
    _test_searchspace_on_dataset(ss, dataset='fake-imagenet')


def test_proxylessnas():
    ss = searchspace.ProxylessNAS()
    _test_searchspace_on_dataset(ss, dataset='fake-imagenet')


def test_mobilenetv3():
    ss = searchspace.MobileNetV3Space()
    _test_searchspace_on_dataset(ss, dataset='fake-imagenet')


def test_shufflenet():
    ss = searchspace.ShuffleNetSpace()
    _test_searchspace_on_dataset(ss, dataset='fake-imagenet')

    ss = searchspace.ShuffleNetSpace(channel_search=True)
    _test_searchspace_on_dataset(ss, dataset='fake-imagenet')


from pytorch_lightning.utilities.seed import seed_everything

# for seed in range(50, 60):
#     print('SEED', seed)
seed_everything(53)
test_nasnet()
