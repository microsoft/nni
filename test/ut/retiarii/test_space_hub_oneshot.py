import nni
import nni.retiarii.hub.pytorch as ss
import nni.retiarii.evaluator.pytorch as pl
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
import nni.retiarii.strategy as strategy
from torchvision import transforms
from torchvision.datasets import CIFAR10


def _hub_factory(alias):
    if alias == 'nasbench101':
        return ss.NasBench101()
    if alias == 'nasbench201':
        return ss.NasBench201()
    if alias.startswith('nasnet'):
        return ss.NASNet(width=16, num_cells=8)
    if alias.startswith('enas'):
        return ss.ENAS(width=16, num_cells=8)
    if alias.startswith('amoeba'):
        return ss.AmoebaNet(width=16, num_cells=8)
    if alias.startswith('pnas'):
        return ss.PNAS(width=16, num_cells=8)
    if alias.startswith('darts'):
        return ss.DARTS(width=16, num_cells=8)



def test_hub_oneshot():
    model_space = ss.ENAS(width=(8, 16), num_cells=(4, 8, 12))

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_loader = pl.DataLoader(
        nni.trace(CIFAR10)('../data/cifar10', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=16,
        shuffle=True
    )

    valid_loader = pl.DataLoader(
        nni.trace(CIFAR10)('../data/cifar10', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=16,
        shuffle=False
    )

    evaluator = pl.Classification(
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        max_epochs=1,
        limit_train_batches=10,
        limit_val_batches=10
    )

    strategy_ = strategy.DARTS(auto_shape_alignment='last')

    config = RetiariiExeConfig()
    config.execution_engine = 'oneshot'
    experiment = RetiariiExperiment(model_space, evaluator, strategy=strategy_)

    experiment.run(config)


# @pytest.mark.parametrize('replace_sampler_ddp', [False, True])
# @pytest.mark.parametrize('is_min_size_mode', [True])
# @pytest.mark.parametrize('num_devices', ['auto', 1, 3, 10])

test_hub_oneshot()