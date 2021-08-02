import json
import logging
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
