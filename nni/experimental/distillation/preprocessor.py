# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from collections import abc
import logging
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Type

from torch.utils.data import DataLoader, Dataset

from .storage import create_storage
from .uid_dataset import create_uid_dataset, _UidDataset
from .utils import _default_labels_split_fn, _default_labels_collate_fn

_logger = logging.getLogger(__name__)


class Preprocessor:
    def __init__(self, teacher_predict: Callable[[Any], Any], dataset: Dataset, create_dataloader: Callable[[Dataset], DataLoader],
                 keep_in_memory: bool = False, cache_folder: str | PathLike | None = None, cache_mode: Literal['pickle', 'hdf5'] = 'pickle',
                 checkpoint_folder: str | PathLike | None = None, uid_dataset_cls: Type[_UidDataset] | None = None,
                 uidd_args: List | None = None, uidd_kwargs: Dict | None = None, labels_split_fn: Callable[[Any], List] | None = None,
                 labels_collate_fn: Callable[[List], Any] | None = None):
        self._storage = create_storage(keep_in_memory=keep_in_memory, cache_folder=cache_folder, cache_mode=cache_mode,
                                       checkpoint_folder=checkpoint_folder)

        self._teacher_predict = teacher_predict
        self._dataset = create_uid_dataset(dataset, uid_dataset_cls, uidd_args=uidd_args, uidd_kwargs=uidd_kwargs)
        self._create_dataloader = create_dataloader

        assert (labels_split_fn is None and labels_collate_fn is None) or (labels_split_fn is not None and labels_collate_fn is not None)
        self._labels_split_fn = labels_split_fn if labels_split_fn is not None else _default_labels_split_fn
        self._labels_collate_fn = labels_collate_fn if labels_collate_fn is not None else _default_labels_collate_fn

        self._preprocessed = False if checkpoint_folder is None else True

    def _patched_uid_dataloader(self):
        dataloader = self._create_dataloader(self._dataset)
        origin_collate_fn = dataloader.collate_fn

        def collate_fn(data):
            uids, origin_data = list(zip(*data))
            return uids, origin_collate_fn(origin_data)

        setattr(dataloader, 'collate_fn', collate_fn)
        return dataloader

    def _patched_label_dataloader(self):
        dataloader = self._patched_uid_dataloader()

        uid_collate_fn = dataloader.collate_fn

        def collate_fn(data):
            uids, origin_batch = uid_collate_fn(data)
            soft_labels = list(map(self._storage.select, uids))
            soft_labels = self._labels_collate_fn(soft_labels)
            return soft_labels, origin_batch

        setattr(dataloader, 'collate_fn', collate_fn)
        return dataloader

    def preprocess_labels(self, epochs: int = 1, log_process: bool = True):
        if self._preprocessed:
            _logger.info('Preprocessor has been preprocessed.')
            return

        self._dataset.observe()

        dataloader = self._patched_uid_dataloader()
        total_batch_num = len(dataloader)
        log_literal = min(max(total_batch_num // 25, 10), 100)

        for epoch_idx in range(epochs):
            for batch_idx, packed_batch in enumerate(dataloader):
                uids, batch = packed_batch
                soft_labels = self._teacher_predict(batch=batch)

                soft_labels = self._labels_split_fn(soft_labels)

                assert isinstance(soft_labels, abc.Iterable)
                list(map(self._storage.record, uids, soft_labels))
                if log_process and ((batch_idx + 1) % log_literal == 0 or (batch_idx + 1) == total_batch_num):
                    info_msg = f'[epoch {epoch_idx}: {batch_idx + 1} / {total_batch_num}]'
                    _logger.info(info_msg)

        # ready for creating replay dataloader
        self._preprocessed = True
        self._dataset.replay()

    def create_replay_dataloader(self):
        if not self._preprocessed:
            raise RuntimeError
        self._dataset.replay()
        return self._patched_label_dataloader()

    def save_checkpoint(self, checkpoint_folder: str | PathLike):
        root_path = Path(checkpoint_folder).absolute()
        root_path.mkdir(parents=True, exist_ok=True)
        self._storage.save_checkpoint(root_path)
