# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import pytest

import torch
from torch.utils.data import Dataset, DataLoader

from nni.experimental.distillation.preprocessor import Preprocessor
from nni.experimental.distillation.uid_dataset import IndexedDataset, HashedDataset, AugmentationDataset

# from ..assets.common import log_dir


class DemoDataset(Dataset):
    def __init__(self) -> None:
        self._data = torch.rand((100, 100))

    def __len__(self):
        return self._data.size()[0]

    def __getitem__(self, index):
        return self._data[index]


def teacher_predict(sample: torch.Tensor):
    return sample.reshape(10, 10, 10).mean(-1)


def create_dataloader(dataset: Dataset):
    return DataLoader(dataset=dataset, batch_size=10)


@pytest.mark.parametrize('uid_dataset_cls', ['index', 'hash', 'augumentation'])
@pytest.mark.parametrize('storage', ['memory', 'file', 'hdf5'])
def test_preprocessor(uid_dataset_cls: str, storage: str):
    dataset = DemoDataset()

    if uid_dataset_cls == 'index':
        uid_dataset = IndexedDataset(dataset=dataset)
    elif uid_dataset_cls == 'hash':
        import hashlib

        def hash(sample: torch.Tensor):
            return hashlib.sha1(sample.numpy()).hexdigest()

        uid_dataset = HashedDataset(dataset=dataset, hash_fn=hash)
    elif uid_dataset_cls == 'augumentation':
        def transform_fn(sample: torch.Tensor):
            new_sample = torch.nn.init.uniform_(sample)
            return new_sample

        uid_dataset = AugmentationDataset(IndexedDataset(dataset=dataset), transform=transform_fn)

    if storage == 'memory':
        preprocessor = Preprocessor(teacher_predict=teacher_predict, dataset=uid_dataset, create_dataloader=create_dataloader,
                                    keep_in_memory=True)
    elif storage == 'file':
        preprocessor = Preprocessor(teacher_predict=teacher_predict, dataset=uid_dataset, create_dataloader=create_dataloader,
                                    keep_in_memory=False, cache_folder='./log', cache_mode='pickle')
    elif storage == 'hdf5':
        preprocessor = Preprocessor(teacher_predict=teacher_predict, dataset=uid_dataset, create_dataloader=create_dataloader,
                                    keep_in_memory=False, cache_folder='./log', cache_mode='hdf5')

    preprocessor.preprocess_labels()
    distil_dataloader = preprocessor.create_replay_dataloader()

    dataloader = create_dataloader(dataset)

    for d1, d2 in zip(dataloader, distil_dataloader):
        assert torch.equal(teacher_predict(d1), d2[0])
