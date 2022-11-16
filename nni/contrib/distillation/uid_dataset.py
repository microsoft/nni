# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Type

import torch
from torch.utils.data import Dataset, IterableDataset

from .utils import _default_get_rngs_state, _default_set_rngs_state, _default_manual_seed


class _UidDataset(Dataset):
    def __init__(self, dataset: Dataset, *args, **kwargs):
        assert not isinstance(dataset, IterableDataset), 'Do not support `IterableDataset`.'
        self._dataset = dataset
        self._replay_mode = False

    def __len__(self):
        return len(self._dataset)  # type: ignore

    def __getitem__(self, index):
        return None, self._dataset.__getitem__(index)

    def get_origin_dataset(self):
        return self._dataset

    def observe(self):
        """
        Observe mode means this dataset is on processing by `Preprocessor`
        """
        self._replay_mode = False

    def replay(self):
        """
        Replay mode means this dataset will replay the previous samples.
        """
        self._replay_mode = True


class IndexedDataset(_UidDataset):
    """
    Return (index, origin_sample).
    """
    def __getitem__(self, index):
        return str(index), self._dataset.__getitem__(index)


class HashedDataset(_UidDataset):
    """
    Return (hash(origin_sample), origin_sample).

    Parameters
    ----------
    hash_fn
        A hash function of sample from dataset, the hash value should be a str.
    """
    def __init__(self, dataset: Dataset, hash_fn: Callable[[Any], str]):
        super().__init__(dataset)
        self._hash_fn = hash_fn

    def __getitem__(self, index):
        sample = self._dataset.__getitem__(index)
        return self._hash_fn(sample), sample


class AugmentationDataset(_UidDataset):
    """
    Return (aux_uid/seed, origin_sample).

    Parameters
    ----------
    dataset
        The original dataset or a `_UidDataset` instance.
        If dataset is a `_UidDataset` instance and `aux_dataset_cls` is None, this dataset will be took as the aux_dataset.
    transform
        The transform function for each sample from dataset.
    seed
        This seed controls the random generators used in transform.
    aux_dataset_cls
        `aux_dataset_cls` is used to wrap the dataset to generate the aux_uid.
    aux_args
        Additional position arguments for `aux_dataset_cls` initialization.
    aux_kwargs
        Additional keyword arguments for `aux_dataset_cls` initialization.
    """
    def __init__(self, dataset: Dataset, transform: Callable[[Any], Any], seed: int | None = None,
                 aux_dataset_cls: Type[_UidDataset] | None = None, *aux_args, **aux_kwargs):
        if isinstance(dataset, _UidDataset) and aux_dataset_cls is None:
            _dataset = dataset
        else:
            aux_dataset_cls = IndexedDataset if aux_dataset_cls is None else aux_dataset_cls
            assert issubclass(aux_dataset_cls, _UidDataset)
            _dataset = aux_dataset_cls(dataset=dataset, *aux_args, **aux_kwargs)
        super().__init__(_dataset)
        self._transform = transform

        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(seed=seed)

        self._replay_mode = False
        self._suid_seed = defaultdict(list)

    def __getitem__(self, index):
        suid, sample = self._dataset.__getitem__(index)

        rngs_state = _default_get_rngs_state()

        if self._replay_mode:
            seed = self._suid_seed[str(suid)].pop(-1)
            self._suid_seed[str(suid)].insert(0, seed)
        else:
            seed = self._generate_seed()
            self._suid_seed[str(suid)].append(seed)
        _default_manual_seed(seed)
        sample = self._transform(sample)

        _default_set_rngs_state(rngs_state)

        return f'{suid}/{seed}', sample

    def _generate_seed(self) -> int:
        return int(torch.randint(-0x8000_0000_0000_0000, 0x7fff_ffff_ffff_ffff, (1,), dtype=torch.long, generator=self._rng).item())

    def get_origin_dataset(self):
        return self._dataset.get_origin_dataset()


def create_uid_dataset(dataset: Dataset, uid_dataset_cls: Type[_UidDataset] | None, uidd_args: List | None, uidd_kwargs: Dict | None):
    if isinstance(dataset, _UidDataset) and uid_dataset_cls is None:
        return dataset
    else:
        uid_dataset_cls = IndexedDataset if uid_dataset_cls is None else uid_dataset_cls
        assert issubclass(uid_dataset_cls, _UidDataset)
        uidd_args = uidd_args if uidd_args is not None else []
        uidd_kwargs = uidd_kwargs if uidd_kwargs is not None else {}
        return uid_dataset_cls(dataset, *uidd_args, **uidd_kwargs)
