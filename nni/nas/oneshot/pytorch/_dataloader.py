# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from typing import Any, Optional

from pytorch_lightning.utilities.combined_loader import (
    CombinedLoader, _CombinationMode, _SUPPORTED_MODES, _Sequential,
    _ModeIterator, _tree_flatten
)

_SUPPORTED_MODES['_nni_concat'] = _CombinationMode(fn=sum, iterator=_Sequential)

__all__ = ['ConcatLoader']


class ConcatLoader(CombinedLoader):
    """This is trying to bypass the supported mode checker in PyTorch-Lightning FitLoop.
    """

    def __init__(self, iterables: Any) -> None:
        self._iterables = iterables
        self._flattened, self._spec = _tree_flatten(iterables)
        self._mode = '_nni_concat'
        self._iterator: Optional[_ModeIterator] = None
