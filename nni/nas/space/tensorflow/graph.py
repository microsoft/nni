# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

__all__ = ['TensorflowGraphModelSpace']

import logging
from typing import ClassVar

from nni.nas.space import GraphModelSpace

_logger = logging.getLogger(__name__)


class TensorflowGraphModelSpace(GraphModelSpace):
    """GraphModelSpace specialized for Tensorflow."""

    framework_type: ClassVar[str] = 'tensorflow'

    def __init__(self, *, _internal=False):
        _logger.warning('Tensorflow model space is not supported yet. It is just a placeholder for internal test purposes.')
        super().__init__(_internal=_internal)
