# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import hashlib
import pytest

import torch
from torch.utils.data import Dataset, DataLoader

from nni.contrib.distillation.label_patcher import DistilLabelPatcher
from nni.contrib.distillation.uid_dataset import IndexedDataset, HashedDataset, AugmentationDataset

from ..assets.common import log_dir


class DemoDataset(Dataset):
    def __init__(self) -> None:
        self._data = torch.rand((100, 100))

    def __len__(self):
        return self._data.size()[0]

    def __getitem__(self, index):
        return self._data[index].clone()


def teacher_predict(sample: torch.Tensor):
    return sample.reshape(10, 10, 10).mean(-1)


def create_dataloader(dataset: Dataset):
    return DataLoader(dataset=dataset, batch_size=10)


def create_augmentation_dataloader(dataset: AugmentationDataset):
    suid_dataset = dataset._dataset

    def collate_fn(data):
        batch = []
        cpu_rng_state = torch.random.get_rng_state()
        for uid, sample in data:
            seed = dataset._suid_seed[str(uid)][-1]
            torch.random.manual_seed(seed)
            batch.append(transform_fn(sample))
        torch.random.set_rng_state(cpu_rng_state)
        return torch.stack(batch)

    return DataLoader(dataset=suid_dataset, batch_size=10, collate_fn=collate_fn)


def transform_fn(sample: torch.Tensor):
    new_sample = torch.nn.init.uniform_(sample)
    return new_sample


def hash(sample: torch.Tensor):
    return hashlib.sha1(sample.numpy()).hexdigest()


@pytest.mark.parametrize('uid_dataset_cls', ['index', 'hash', 'augumentation'])
@pytest.mark.parametrize('storage', ['memory', 'file'])
def test_distil_label_patcher(uid_dataset_cls: str, storage: str):
    dataset = DemoDataset()

    if uid_dataset_cls == 'index':
        uid_dataset = IndexedDataset(dataset=dataset)
    elif uid_dataset_cls == 'hash':
        uid_dataset = HashedDataset(dataset=dataset, hash_fn=hash)
    elif uid_dataset_cls == 'augumentation':
        uid_dataset = AugmentationDataset(IndexedDataset(dataset=dataset), transform=transform_fn)

    if storage == 'memory':
        patcher = DistilLabelPatcher(teacher_predict=teacher_predict, dataset=uid_dataset, create_dataloader=create_dataloader,
                                     keep_in_memory=True)
    elif storage == 'file':
        patcher = DistilLabelPatcher(teacher_predict=teacher_predict, dataset=uid_dataset, create_dataloader=create_dataloader,
                                     keep_in_memory=False, cache_folder=log_dir, cache_mode='pickle')
    # add this storage back after it is done
    # elif storage == 'hdf5':
    #     patcher = DistilLabelPatcher(teacher_predict=teacher_predict, dataset=uid_dataset, create_dataloader=create_dataloader,
    #                                  keep_in_memory=False, cache_folder=log_dir, cache_mode='hdf5')

    patcher.generate_distillation_labels()
    distil_dataloader = patcher.create_patched_dataloader()

    if not uid_dataset_cls == 'augumentation':
        dataloader = create_dataloader(dataset)
        for d1, d2 in zip(dataloader, distil_dataloader):
            assert torch.equal(teacher_predict(d1), d2[0])
    else:
        dataloader = create_augmentation_dataloader(uid_dataset)
        for d1, d2 in zip(dataloader, distil_dataloader):
            assert torch.equal(teacher_predict(d1), d2[0])
