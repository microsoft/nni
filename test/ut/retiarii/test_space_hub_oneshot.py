import nni
import nni.retiarii.hub.pytorch as ss
import nni.retiarii.evaluator.pytorch as pl
import nni.retiarii.strategy as stg
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
from nni.retiarii.hub.pytorch.nasnet import NDSStagePathSampling, NDSStageDifferentiable
from torchvision import transforms
from torchvision.datasets import CIFAR10


def _hub_factory(alias):
    if alias == 'nasbench101':
        return ss.NasBench101()
    if alias == 'nasbench201':
        return ss.NasBench201()

    if alias == 'mobilenetv3':
        return ss.MobileNetV3Space()
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

    if alias.startswith('nasnet'):
        return ss.NASNet(width=width, num_cells=num_cells)
    if alias.startswith('enas'):
        return ss.ENAS(width=width, num_cells=num_cells)
    if alias.startswith('amoeba'):
        return ss.AmoebaNet(width=width, num_cells=num_cells)
    if alias.startswith('pnas'):
        return ss.PNAS(width=width, num_cells=num_cells)
    if alias.startswith('darts'):
        return ss.DARTS(width=width, num_cells=num_cells)

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
        return stg.ENAS(mutation_hooks=extra_mutation_hooks)
    if alias == 'random':
        return stg.RandomOneShot(mutation_hooks=extra_mutation_hooks)

    raise ValueError(f'Unrecognized strategy: {alias}')


def test_hub_oneshot(space_type, strategy_type):
    model_space = _hub_factory(space_type)

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_loader = pl.DataLoader(
        nni.trace(CIFAR10)('../data/cifar10', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=4,
        shuffle=True
    )

    valid_loader = pl.DataLoader(
        nni.trace(CIFAR10)('../data/cifar10', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=4,
        shuffle=False
    )

    evaluator = pl.Classification(
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        max_epochs=1,
        limit_train_batches=50,
        limit_val_batches=50
    )

    strategy = _strategy_factory(strategy_type, space_type)

    config = RetiariiExeConfig()
    config.execution_engine = 'oneshot'
    experiment = RetiariiExperiment(model_space, evaluator, strategy=strategy)

    experiment.run(config)


# @pytest.mark.parametrize('replace_sampler_ddp', [False, True])
# @pytest.mark.parametrize('is_min_size_mode', [True])
# @pytest.mark.parametrize('num_devices', ['auto', 1, 3, 10])

test_hub_oneshot('enas_smalldepth_width', 'darts')