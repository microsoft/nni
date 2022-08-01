import logging
import pytest

import numpy as np
import torch

import nni
import nni.retiarii.hub.pytorch as ss
import nni.retiarii.evaluator.pytorch as pl
import nni.retiarii.strategy as stg
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
from nni.retiarii.hub.pytorch.nasnet import NDSStagePathSampling, NDSStageDifferentiable
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageNet

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='Too slow without CUDA.')


def _hub_factory(alias):
    if alias == 'nasbench101':
        return ss.NasBench101()
    if alias == 'nasbench201':
        return ss.NasBench201()

    if alias == 'mobilenetv3':
        return ss.MobileNetV3Space()

    if alias == 'mobilenetv3_small':
        return ss.MobileNetV3Space(
            width_multipliers=(0.75, 1, 1.5),
            expand_ratios=(4, 6)
        )
    if alias == 'proxylessnas':
        return ss.ProxylessNAS()
    if alias == 'shufflenet':
        return ss.ShuffleNetSpace()
    if alias == 'autoformer':
        return ss.AutoformerSpace()

    if '_smalldepth' in alias:
        num_cells = (4, 8)
    elif '_depth' in alias:
        num_cells = (8, 12)
    else:
        num_cells = 8

    if '_width' in alias:
        width = (8, 16)
    else:
        width = 16

    if '_imagenet' in alias:
        dataset = 'imagenet'
    else:
        dataset = 'cifar'

    if alias.startswith('nasnet'):
        return ss.NASNet(width=width, num_cells=num_cells, dataset=dataset)
    if alias.startswith('enas'):
        return ss.ENAS(width=width, num_cells=num_cells, dataset=dataset)
    if alias.startswith('amoeba'):
        return ss.AmoebaNet(width=width, num_cells=num_cells, dataset=dataset)
    if alias.startswith('pnas'):
        return ss.PNAS(width=width, num_cells=num_cells, dataset=dataset)
    if alias.startswith('darts'):
        return ss.DARTS(width=width, num_cells=num_cells, dataset=dataset)

    raise ValueError(f'Unrecognized space: {alias}')


def _strategy_factory(alias, space_type):
    # Some search space needs extra hooks
    extra_mutation_hooks = []
    nds_need_shape_alignment = '_smalldepth' in space_type
    if nds_need_shape_alignment:
        if alias in ['enas', 'random']:
            extra_mutation_hooks.append(NDSStagePathSampling.mutate)
        else:
            extra_mutation_hooks.append(NDSStageDifferentiable.mutate)

    if alias == 'darts':
        return stg.DARTS(mutation_hooks=extra_mutation_hooks)
    if alias == 'gumbel':
        return stg.GumbelDARTS(mutation_hooks=extra_mutation_hooks)
    if alias == 'proxyless':
        return stg.Proxyless()
    if alias == 'enas':
        return stg.ENAS(mutation_hooks=extra_mutation_hooks, reward_metric_name='val_acc')
    if alias == 'random':
        return stg.RandomOneShot(mutation_hooks=extra_mutation_hooks)

    raise ValueError(f'Unrecognized strategy: {alias}')


def _dataset_factory(dataset_type, subset=20):
    if dataset_type == 'cifar10':
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_dataset = nni.trace(CIFAR10)(
            'data/cifar10',
            train=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]))
        valid_dataset = nni.trace(CIFAR10)(
            'data/cifar10',
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
    elif dataset_type == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_dataset = nni.trace(ImageNet)(
            'data/imagenet',
            split='val',  # no train data available in tests
            transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        valid_dataset = nni.trace(ImageNet)(
            'data/imagenet',
            split='val',
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
    else:
        raise ValueError(f'Unsupported dataset type: {dataset_type}')

    if subset:
        train_dataset = Subset(train_dataset, np.random.permutation(len(train_dataset))[:subset])
        valid_dataset = Subset(valid_dataset, np.random.permutation(len(valid_dataset))[:subset])

    return train_dataset, valid_dataset


@pytest.mark.parametrize('space_type', [
    # 'nasbench101',
    'nasbench201',
    'mobilenetv3',
    'mobilenetv3_small',
    'proxylessnas',
    'shufflenet',
    # 'autoformer',
    'nasnet',
    'enas',
    'amoeba',
    'pnas',
    'darts',

    'darts_smalldepth',
    'darts_depth',
    'darts_width',
    'darts_width_smalldepth',
    'darts_width_depth',
    'darts_imagenet',
    'darts_width_smalldepth_imagenet',

    'enas_smalldepth',
    'enas_depth',
    'enas_width',
    'enas_width_smalldepth',
    'enas_width_depth',
    'enas_imagenet',
    'enas_width_smalldepth_imagenet',

    'pnas_width_smalldepth',
    'amoeba_width_smalldepth',
])
@pytest.mark.parametrize('strategy_type', [
    'darts',
    'gumbel',
    'proxyless',
    'enas',
    'random'
])
def test_hub_oneshot(space_type, strategy_type):
    NDS_SPACES = ['amoeba', 'darts', 'pnas', 'enas', 'nasnet']
    if strategy_type == 'proxyless':
        if 'width' in space_type or 'depth' in space_type or \
                any(space_type.startswith(prefix) for prefix in NDS_SPACES + ['proxylessnas', 'mobilenetv3']):
            pytest.skip('The space has used unsupported APIs.')
    if strategy_type in ['darts', 'gumbel'] and space_type == 'mobilenetv3':
        pytest.skip('Skip as it consumes too much memory.')

    model_space = _hub_factory(space_type)

    dataset_type = 'cifar10'
    if 'imagenet' in space_type or space_type in ['mobilenetv3', 'proxylessnas', 'shufflenet', 'autoformer']:
        dataset_type = 'imagenet'

    subset_size = 4
    if strategy_type in ['darts', 'gumbel'] and any(space_type.startswith(prefix) for prefix in NDS_SPACES) and '_' in space_type:
        subset_size = 2

    train_dataset, valid_dataset = _dataset_factory(dataset_type, subset=subset_size)
    train_loader = pl.DataLoader(train_dataset, batch_size=2, num_workers=2, shuffle=True)
    valid_loader = pl.DataLoader(valid_dataset, batch_size=2, num_workers=2, shuffle=False)

    evaluator = pl.Classification(
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        max_epochs=1,
        export_onnx=False,
        gpus=1 if torch.cuda.is_available() else 0,  # 0 for my debug
        logger=False,  # disable logging and checkpoint to avoid too much log
        enable_checkpointing=False,
        enable_model_summary=False
        # profiler='advanced'
    )

    # To test on final model:
    # model = type(model_space).load_searched_model('darts-v2')
    # evaluator.fit(model)

    strategy = _strategy_factory(strategy_type, space_type)

    config = RetiariiExeConfig()
    config.execution_engine = 'oneshot'
    experiment = RetiariiExperiment(model_space, evaluator, strategy=strategy)

    experiment.run(config)


_original_loglevel = None

def setup_module(module):
    global _original_loglevel
    _original_loglevel = logging.getLogger("pytorch_lightning").level
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


def teardown_module(module):
    logging.getLogger("pytorch_lightning").setLevel(_original_loglevel)
