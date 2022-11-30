# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from collections import abc
import logging
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Type, cast

from torch.utils.data import DataLoader, Dataset

from .storage import create_storage
from .uid_dataset import create_uid_dataset, _UidDataset
from .utils import _default_labels_split_fn, _default_labels_collate_fn

_logger = logging.getLogger(__name__)


class DistilLabelPatcher:
    """
    DistilLabelPatcher is a utility class to generate offline distillation dataloader.

    Parameters
    ----------
    teacher_predict
        The teacher predict function, given an input and return the logits/predictions/data that distillation needs.
        Usually, it could be an evaluation mode model.
    dataset
        The training dataset, it will be wrapped by a `_UidDataset` in this patcher.
        If this dataset is already an instance of `_UidDataset` and `uid_dataset_cls` is None, it will not be wrap again.
        (Reference to what is _UidDataset ...)
    create_dataloader
        A factory function to create a dataloader for the given `dataset`.
    keep_in_memory
        By default, False. If using memory or disk to save the distillation data (`teacher_predict` outputs).
    cache_folder
        By default, None. The path to save the data if `keep_in_memory` is False. If `cache_folder` is None,
        a temp folder will be created to save data.
        This argument will be ignored if `keep_in_memory` is True.
    cache_mode
        (Not stable!)  By default, `pickle`. Using `pickle` or `hdf5` to save data.
    checkpoint_folder
        By default, None. The path to load storage state to recover the previous saved distillation data.
    uid_dataset_cls
        By default, None. If `dataset` is not an instance of `_UidDataset` and `uid_dataset_cls` is None,
        `IndexedDataset` will be used to wrap `dataset`. `IndexedDataset` take the index passed to `Dataset.__getitem__` as data's uid.
        `uid_dataset_cls` should be a subclass of `_UidDataset`, `uid_dataset_cls.__getitem__` will return a tuple of (uid, original_data).
        (Official uid dataset and how to customize uid dataset, please refer ...)
    uidd_args
        Additional arguments pass to `uid_dataset_cls.__init__`.
        If there are many extra arguments, recommended to directly set `uid_dataset_cls` to None, and then pass in the wrapped `dataset`.
    uidd_kwargs
        Additional keyword arguments pass to `uid_dataset_cls.__init__`.
        If there are many extra arguments, recommended to directly set `uid_dataset_cls` to None, and then pass in the wrapped `dataset`.
    labels_split_fn
        Please refer `DistilLabelPatcher.generate_distillation_labels`.
    labels_collate_fn
        Please refer `DistilLabelPatcher.create_patched_dataloader`.
    """
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

        self._label_done = False if checkpoint_folder is None else True

    def _patched_uid_dataloader(self):
        dataloader = self._create_dataloader(self._dataset)
        origin_collate_fn = dataloader.collate_fn

        def collate_fn(data):
            uids, origin_data = list(zip(*data))
            return uids, origin_collate_fn(origin_data)  # type: ignore

        setattr(dataloader, 'collate_fn', collate_fn)
        return dataloader

    def _patched_label_dataloader(self, labels_collate_fn: Callable[[List], Any]):
        dataloader = self._patched_uid_dataloader()

        uid_collate_fn = dataloader.collate_fn

        def collate_fn(data):
            uids, origin_batch = uid_collate_fn(data)
            distil_labels = list(map(self._storage.select, cast(List[str], uids)))
            distil_labels = labels_collate_fn(distil_labels)
            return distil_labels, origin_batch

        setattr(dataloader, 'collate_fn', collate_fn)
        return dataloader

    def generate_distillation_labels(self, epochs: int = 1, log_process: bool = True, labels_split_fn: Callable[[Any], List] | None = None):
        """
        Run the `teacher_predict` on the whole dataset, and record the output as distillation labels.

        Parameters
        ----------
        epochs
            By default, 1. It specifies how many epochs used to generate distillation labels, usually it can be set to 1.
            And in some training methods, such as data augumentation, each time the same meta data selected from dataset
            might be different after augumentation.
            At this time, setting `>1` can run more epochs to obtain multiple epochs of distillation labels.
        log_process
            If logging during generate labels.
        labels_split_fn
            A function used to split the batched output of `teacher_predict` to list of output. If `labels_split_fn` is None,
            `teacher_predict` is expected to return a batched tensor, and it will be auto split on dim 0 by this function.
            For example, `teacher_predict` returns a tensor with size (32, 100), which is the probability of 100 classes for 32 samples,
            then after `labels_split_fn`, it will be split to a list of tensor with 32 elements,
            each tensor is the probability with size (100,).
        """
        if self._label_done:
            warn_msg = 'DistilLabelPatcher has been generated distillation labels, if more epochs are needed, please set '
            _logger.warning(warn_msg)
            return

        labels_split_fn = labels_split_fn if labels_split_fn is not None else self._labels_split_fn

        self._dataset.observe()

        dataloader = self._patched_uid_dataloader()
        total_batch_num = len(dataloader)
        log_literal = min(max(total_batch_num // 25, 10), 100)

        for epoch_idx in range(epochs):
            for batch_idx, packed_batch in enumerate(dataloader):
                uids, batch = packed_batch
                soft_labels = self._teacher_predict(batch)

                soft_labels = labels_split_fn(soft_labels)

                assert isinstance(soft_labels, abc.Iterable)
                list(map(self._storage.record, uids, soft_labels))
                if log_process and ((batch_idx + 1) % log_literal == 0 or (batch_idx + 1) == total_batch_num):
                    info_msg = f'[epoch {epoch_idx}: {batch_idx + 1} / {total_batch_num}]'
                    _logger.info(info_msg)

        # ready for creating patched dataloader
        self._label_done = True
        self._dataset.replay()

    def create_patched_dataloader(self, labels_collate_fn: Callable[[List], Any] | None = None):
        """
        Return a dataloader that concat the original dataset and the generated distillation labels.
        The iteration of dataloader will return (distil_labels, origin_batch).

        Parameters
        ----------
        labels_collate_fn
            Use to create patched dataloader.
            A function used to collate the samples from distillation storage. If `labels_collate_fn` is None,
            the samples are excepted to be tensors and have same size, they will be batched and return.
        """
        labels_collate_fn = labels_collate_fn if labels_collate_fn is not None else self._labels_collate_fn
        if not self._label_done:
            raise RuntimeError
        self._dataset.replay()
        return self._patched_label_dataloader(labels_collate_fn)

    def save_checkpoint(self, checkpoint_folder: str | PathLike):
        """
        Save a checkpoint that can be used to re-initialize the `DistilLabelPatcher` and reload the generated distillation labels.

        Parameters
        ----------
        checkpoint_folder
            The directory to save checkpoint.
        """
        root_path = Path(checkpoint_folder).absolute()
        root_path.mkdir(parents=True, exist_ok=True)
        self._storage.save_checkpoint(root_path)
