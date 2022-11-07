# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Union, Dict, Any

from .utils import ContextStack

_logger = logging.getLogger(__name__)


def fixed_arch(fixed_arch: Union[str, Path, Dict[str, Any]], verbose=True):
    """
    Load architecture from ``fixed_arch`` and apply to model. This should be used as a context manager. For example,

    .. code-block:: python

        with fixed_arch('/path/to/export.json'):
            model = Model(3, 224, 224)

    Parameters
    ----------
    fixed_arc : str, Path or dict
        Path to the JSON that stores the architecture, or dict that stores the exported architecture.
    verbose : bool
        Print log messages if set to True

    Returns
    -------
    ContextStack
        Context manager that provides a fixed architecture when creates the model.
    """

    if isinstance(fixed_arch, (str, Path)):
        with open(fixed_arch) as f:
            fixed_arch = json.load(f)

    if verbose:
        _logger.info(f'Fixed architecture: %s', fixed_arch)

    return ContextStack('fixed', fixed_arch)


@contextmanager
def no_fixed_arch():
    """
    Ignore the ``fixed_arch()`` context.

    This is useful in creating a search space within a ``fixed_arch()`` context.
    Under the hood, it only disables the most recent one fixed context, which means,
    if it's currently in a nested with-fixed-arch context, multiple ``no_fixed_arch()`` contexts is required.

    Examples
    --------
    >>> with fixed_arch(arch_dict):
    ...     with no_fixed_arch():
    ...         model_space = ModelSpace()
    """

    NO_ARCH = '_no_arch_'

    popped_arch = NO_ARCH  # make linter happy
    try:
        try:
            popped_arch = ContextStack.pop('fixed')
        except IndexError:
            # context unavailable
            popped_arch = NO_ARCH
        yield
    finally:
        if popped_arch is not NO_ARCH:
            ContextStack.push('fixed', popped_arch)
