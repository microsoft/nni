# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

__all__ = ['current_model', 'model_context']

import copy
from typing import Optional
from nni.mutable import frozen_context, Sample


def current_model() -> Optional[Sample]:
    """Get the current model sample in :func:`model_context`.

    The sample is supposed to be the same as :attr:`nni.nas.space.ExecutableModelSpace.sample`.

    This method is only valid when called inside :func:`model_context`.
    By default, only the execution of :class:`~nni.nas.space.SimplifiedModelSpace` will set the context,
    so that :func:`current_model` is meaningful within the re-instantiation of the model.

    Returns
    -------
    Model sample (i.e., architecture dict) before freezing, produced by strategy.
    If not called inside :func:`model_context`, returns None.
    """
    cur = frozen_context.current()
    if cur is None or not cur.get('__arch__'):
        # frozen_context exists but it's not set by arch.
        return None
    cur = copy.copy(cur)
    cur.pop('__arch__')
    return cur


def model_context(sample: Sample) -> frozen_context:
    """Get a context stack of the current model sample (i.e., architecture dict).

    This should be used together with :func:`current_model`.

    :func:`model_context` is read-only, and should not be used to modify the architecture dict.
    """
    return frozen_context({**sample, '__arch__': True})
