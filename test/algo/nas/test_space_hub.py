"""Currently, this is only a sanity-check (runnable) of spaces provided in hub."""

import random

from torchvision import transforms
from torchvision.datasets import FakeData

import pytest

import pytorch_lightning

import nni
import nni.runtime.platform.test
import nni.retiarii.evaluator.pytorch.lightning as pl
import nni.retiarii.hub.pytorch as searchspace
from nni.retiarii import fixed_arch
from nni.retiarii.execution.utils import unpack_if_only_one
from nni.retiarii.mutator import InvalidMutation, Sampler
from nni.retiarii.nn.pytorch.mutator import extract_mutation_from_pt_module


pytestmark = pytest.mark.skipif(pytorch_lightning.__version__ < '1.0', reason='Incompatible APIs.')


def _reset():
    # this is to not affect other tests in sdk
    nni.trial._intermediate_seq = 0
    nni.trial._params = {'foo': 'bar', 'parameter_id': 0, 'parameters': {}}
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


def _test_searchspace_on_dataset(searchspace, dataset='cifar10', arch=None):
    _reset()

    # dataset supports cifar10 and imagenet
    model, mutators = extract_mutation_from_pt_module(searchspace)

    if arch is None:
        model = try_mutation_until_success(model, mutators, 10)
        arch = {mut.mutator.label: unpack_if_only_one(mut.samples) for mut in model.history}

    print('Selected model:', arch)
    with fixed_arch(arch):
        model = model.python_class(**model.python_init_params)

    if dataset == 'cifar10':
        train_data = FakeData(size=200, image_size=(3, 32, 32), num_classes=10, transform=transforms.ToTensor())
        valid_data = FakeData(size=200, image_size=(3, 32, 32), num_classes=10, transform=transforms.ToTensor())

    elif dataset == 'imagenet':
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

    # cleanup to avoid affecting later test cases
    _reset()


def test_nasbench101():
    ss = searchspace.NasBench101()
    _test_searchspace_on_dataset(ss)


def test_nasbench201():
    ss = searchspace.NasBench201()
    _test_searchspace_on_dataset(ss)


def test_nasnet():
    _test_searchspace_on_dataset(searchspace.NASNet())
    _test_searchspace_on_dataset(searchspace.ENAS())
    _test_searchspace_on_dataset(searchspace.AmoebaNet())
    _test_searchspace_on_dataset(searchspace.PNAS())
    _test_searchspace_on_dataset(searchspace.DARTS())


def test_nasnet_corner_case():
    # The case is that output channel of reduce cell and normal cell are different
    # CellPreprocessor needs to know whether its predecessors are normal cell / reduction cell
    arch = {
        "width": 32,
        "depth": 8,
        "normal/op_2_0": "max_pool_7x7",
        "normal/op_2_1": "conv_1x1",
        "normal/op_3_0": "sep_conv_5x5",
        "normal/op_3_1": "max_pool_7x7",
        "normal/op_4_0": "sep_conv_5x5",
        "normal/op_4_1": "conv_1x1",
        "normal/op_5_0": "max_pool_3x3",
        "normal/op_5_1": "sep_conv_5x5",
        "normal/op_6_0": "max_pool_7x7",
        "normal/op_6_1": "sep_conv_5x5",
        "normal/input_2_0": 0,
        "normal/input_2_1": 0,
        "normal/input_3_0": 0,
        "normal/input_3_1": 1,
        "normal/input_4_0": 1,
        "normal/input_4_1": 2,
        "normal/input_5_0": 0,
        "normal/input_5_1": 1,
        "normal/input_6_0": 0,
        "normal/input_6_1": 2,
        "reduce/op_2_0": "dil_conv_3x3",
        "reduce/op_2_1": "max_pool_7x7",
        "reduce/op_3_0": "dil_conv_3x3",
        "reduce/op_3_1": "dil_conv_3x3",
        "reduce/op_4_0": "conv_7x1_1x7",
        "reduce/op_4_1": "conv_7x1_1x7",
        "reduce/op_5_0": "max_pool_3x3",
        "reduce/op_5_1": "conv_1x1",
        "reduce/op_6_0": "sep_conv_7x7",
        "reduce/op_6_1": "sep_conv_3x3",
        "reduce/input_2_0": 1,
        "reduce/input_2_1": 1,
        "reduce/input_3_0": 0,
        "reduce/input_3_1": 1,
        "reduce/input_4_0": 2,
        "reduce/input_4_1": 1,
        "reduce/input_5_0": 0,
        "reduce/input_5_1": 4,
        "reduce/input_6_0": 3,
        "reduce/input_6_1": 3,
    }

    _test_searchspace_on_dataset(searchspace.NASNet(), arch=arch)


def test_nasnet_fixwd():
    # minimum
    ss = searchspace.DARTS(width=16, num_cells=4)
    _test_searchspace_on_dataset(ss)

    # medium
    ss = searchspace.NASNet(width=16, num_cells=12)
    _test_searchspace_on_dataset(ss)


def test_nasnet_imagenet():
    ss = searchspace.ENAS(dataset='imagenet')
    _test_searchspace_on_dataset(ss, dataset='imagenet')

    ss = searchspace.PNAS(dataset='imagenet')
    _test_searchspace_on_dataset(ss, dataset='imagenet')


def test_proxylessnas():
    ss = searchspace.ProxylessNAS()
    _test_searchspace_on_dataset(ss, dataset='imagenet')


def test_mobilenetv3():
    ss = searchspace.MobileNetV3Space()
    _test_searchspace_on_dataset(ss, dataset='imagenet')


def test_shufflenet():
    ss = searchspace.ShuffleNetSpace()
    _test_searchspace_on_dataset(ss, dataset='imagenet')

    ss = searchspace.ShuffleNetSpace(channel_search=True)
    _test_searchspace_on_dataset(ss, dataset='imagenet')

def test_autoformer():
    ss = searchspace.AutoformerSpace()
    _test_searchspace_on_dataset(ss, dataset='imagenet')
