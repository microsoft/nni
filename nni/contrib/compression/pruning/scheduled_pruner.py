# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from ..base.compressor import Pruner


class ScheduledPruner(Pruner):
    def __init__(self, *args, **kwargs):
        raise RuntimeError(f'please using {self.__class__.__name__}.')

    @classmethod
    def from_compressor(cls, compressor: Pruner):
        assert isinstance(compressor, Pruner)
        return super().from_compressor(compressor, [])
