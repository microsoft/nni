import torch
from torch.utils.data import DataLoader
from nni.retiarii.oneshot.pytorch.utils import ConcatLoader


def test_concat_loader():
    loaders = {
        "a": DataLoader(range(10), batch_size=4),
        "b": DataLoader(range(20), batch_size=5),
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
    loaders = {
        "a": [DataLoader(range(10), batch_size=4), DataLoader(range(20), batch_size=6)],
        "b": DataLoader(range(20), batch_size=5),
    }
    dataloader = ConcatLoader(loaders)
    assert len(dataloader) == 7
    for i, (data, label) in enumerate(dataloader):
        if i < 3:
            assert isinstance(data, list) and len(data) == 2
            assert label == 'a'
        else:
            assert label == 'b'
