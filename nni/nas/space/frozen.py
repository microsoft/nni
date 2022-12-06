# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

__all__ = ['current_model', 'model_context']

import copy
from typing import Optional, Dict, Any
from nni.mutable import frozen_context, ContextStack


def current_model() -> Optional[dict]:
    """Get the current architecture dict in :func:`model_context`.

    This is only valid when called inside :func:`model_context`.
    By default, only when :class:`~nni.nas.space.SimplifiedModelSpace` is used as the model format,
    :func:`current_model` is meaningful.

    Returns
    -------
    Architecture dict before freezing, produced by strategy.
    If not called inside :func:`model_context`, returns None.
    """
    cur = frozen_context.current()
    if cur is None or not cur.get('__arch__'):
        # frozen_context exists but it's not set by arch.
        return None
    cur = copy.copy(cur)
    cur.pop('__arch__')
    return cur


def model_context(arch: Dict[str, Any]) -> ContextStack:
    """Get a context stack of the current architecture.

    This should be used together with :func:`current_model`.
    """
    return frozen_context({**arch, '__arch__': True})
