# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# pylint: skip-file
# type: ignore

from __future__ import annotations

from typing import Any

from pytorch_lightning.trainer.supporters import CombinedLoader, CombinedLoaderIterator

__all__ = ['ConcatLoader']


class ConcatLoader(CombinedLoader):
    """This loader is same as CombinedLoader in PyTorch-Lightning, but concatenate sub-loaders
    instead of loading them in parallel.

    Parameters
    ----------
    loaders
        For example, ::

            {
                "train": DataLoader(train_dataset),
                "val": DataLoader(val_dataset)
            }

        In this example, the loader will first produce the batches from "train", then "val".

    mode
        Only support "min_size" for now.
    """

    def __init__(self, loaders: dict[str, Any], mode: str = 'min_size'):
        # FIXME: max_cycle will make dataloaders cycle iterators,
        # causing extra problems.
        if mode != 'min_size':
            raise ValueError('Only min_size mode is supported now.')
        super().__init__(loaders, mode)

    def __iter__(self) -> Any:
        """Replace the super-class iterator with ours."""
        self._try_to_patch_pytorch_dataloader()
        iterator = ConcatLoaderIterator(self.loaders)
        # handle fault tolerant restart.
        self.on_restart(iterator)
        self._iterator = iterator
        return iterator

    @staticmethod
    def _try_to_patch_pytorch_dataloader():
        """Copied from CombinedLoader."""
        from torch.utils.data.dataloader import _BaseDataLoaderIter

        # prevent `NotImplementedError` from PyTorch:
        # https://github.com/pytorch/pytorch/blob/v1.9.0/torch/utils/data/dataloader.py#L541
        def __getstate__patch__(*_):
            return {}

        _BaseDataLoaderIter.__getstate__ = __getstate__patch__  # type: ignore

    def __len__(self) -> int:
        return int(sum(self._calc_num_batches(loader) for loader in self.loaders.values()))


class ConcatLoaderIterator(CombinedLoaderIterator):
    """Similar to CombinedLoaderIterator in Lightning, but in a concat manner."""

    def __next__(self) -> Any:
        """Fetches the next batch from multiple data loaders,
        by looking for the first iterator that isn't exhausted yet.
        """
        if not len(self.loader_iters) == len(self.loaders):
            raise RuntimeError('loader_iters must have the same length as loaders.')
        for i, (loader_name, iterator) in enumerate(self.loader_iters.items()):
            try:
                return (self.request_next_batch(iterator), loader_name)
            except StopIteration:
                if i + 1 == len(self.loader_iters):
                    raise
