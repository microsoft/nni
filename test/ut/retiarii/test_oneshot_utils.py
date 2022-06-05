import math
from typing import Union

import pytest
import torch
import pytorch_lightning
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, Dataset

pytestmark = pytest.mark.skipif(pytorch_lightning.__version__ < '1.0', reason='Incompatible APIs')


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log('valid_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)



def test_concat_loader():
    from nni.retiarii.oneshot.pytorch.dataloader import ConcatLoader

    loaders = {
        'a': DataLoader(range(10), batch_size=4),
        'b': DataLoader(range(20), batch_size=5),
    }
    dataloader = ConcatLoader(loaders)
    assert len(dataloader) == 7
    for i, (data, label) in enumerate(dataloader):
        if i < 3:
            assert len(data) <= 4
            assert label == 'a'
        else:
            assert len(data) <= 5
            assert label == 'b'


def test_concat_loader_nested():
    from nni.retiarii.oneshot.pytorch.dataloader import ConcatLoader

    loaders = {
        'a': [DataLoader(range(10), batch_size=4), DataLoader(range(20), batch_size=6)],
        'b': DataLoader(range(20), batch_size=5),
    }
    dataloader = ConcatLoader(loaders)
    assert len(dataloader) == 7
    for i, (data, label) in enumerate(dataloader):
        if i < 3:
            assert isinstance(data, list) and len(data) == 2
            assert label == 'a'
        else:
            assert label == 'b'


@pytest.mark.parametrize('replace_sampler_ddp', [False, True])
@pytest.mark.parametrize('is_min_size_mode', [True])
@pytest.mark.parametrize('num_devices', ['auto', 1, 3, 10])
def test_concat_loader_with_ddp(
    replace_sampler_ddp: bool, is_min_size_mode: bool, num_devices: Union[int, str]
):
    """Inspired by tests/trainer/test_supporters.py in lightning."""
    from nni.retiarii.oneshot.pytorch.dataloader import ConcatLoader

    mode = 'min_size' if is_min_size_mode else 'max_size_cycle'
    dim = 3
    n1 = 8
    n2 = 6
    n3 = 9
    dataloader = ConcatLoader({
        'a': {
            'a1': DataLoader(RandomDataset(dim, n1), batch_size=1),
            'a2': DataLoader(RandomDataset(dim, n2), batch_size=1),
        },
        'b': DataLoader(RandomDataset(dim, n3), batch_size=1),
    }, mode=mode)
    expected_length_before_ddp = n3 + (min(n1, n2) if is_min_size_mode else max(n1, n2))
    print(len(dataloader))
    assert len(dataloader) == expected_length_before_ddp
    model = BoringModel()
    trainer = Trainer(
        strategy='ddp',
        accelerator='cpu',
        devices=num_devices,
        replace_sampler_ddp=replace_sampler_ddp,
    )
    trainer._data_connector.attach_data(
        model=model, train_dataloaders=dataloader, val_dataloaders=None, datamodule=None
    )
    expected_length_after_ddp = (
        math.ceil(n3 / trainer.num_devices) + \
            math.ceil((min(n1, n2) if is_min_size_mode else max(n1, n2)) / trainer.num_devices)
        if replace_sampler_ddp
        else expected_length_before_ddp
    )
    print('Num devices =', trainer.num_devices)
    trainer.reset_train_dataloader(model=model)
    assert trainer.train_dataloader is not None
    assert trainer.train_dataloader.mode == mode
    
    assert trainer.num_training_batches == expected_length_after_ddp
