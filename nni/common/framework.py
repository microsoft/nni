# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

__all__ = ['set_default_framework', 'get_default_framework', 'shortcut_module', 'shortcut_framework']

import importlib
import os
import sys
from typing import Optional, cast

from typing_extensions import Literal

framework_type = Literal['pytorch', 'tensorflow', 'mxnet', 'none']
"""Supported framework types."""

ENV_NNI_FRAMEWORK = 'NNI_FRAMEWORK'

def framework_from_env() -> framework_type:
    framework = os.getenv(ENV_NNI_FRAMEWORK, 'pytorch')
    if framework not in framework_type.__args__:  # type: ignore
        raise ValueError(f'{framework} does not belong to {framework_type.__args__}')  # type: ignore
    return cast(framework_type, framework)


DEFAULT_FRAMEWORK = framework_from_env()


def set_default_framework(framework: framework_type) -> None:
    """Set default deep learning framework to simplify imports.

    Some functionalities in NNI (e.g., NAS / Compression), relies on an underlying DL framework.
    For different DL frameworks, the implementation of NNI can be very different.
    Thus, users need import things tailored for their own framework. For example: ::

        from nni.nas.xxx.pytorch import yyy

    rather than: ::

        from nni.nas.xxx import yyy

    By setting a default framework, shortcuts will be made. As such ``nni.nas.xxx`` will be equivalent to ``nni.nas.xxx.pytorch``.

    Another way to setting it is through environment variable ``NNI_FRAMEWORK``,
    which needs to be set before the whole process starts.

    If you set the framework with :func:`set_default_framework`,
    it should be done before all imports (except nni itself) happen,
    because it will affect other import's behaviors.
    And the behavior is undefined if the framework is "re"-set in the middle.

    The supported frameworks here are listed below.
    It doesn't mean that they are fully supported by NAS / Compression in NNI.

    * ``pytorch`` (default)
    * ``tensorflow``
    * ``mxnet``
    * ``none`` (to disable the shortcut-import behavior).

    Examples
    --------
    >>> import nni
    >>> nni.set_default_framework('tensorflow')
    >>> # then other imports
    >>> from nni.nas.xxx import yyy
    """

    # In case 'none' is written as None.
    if framework is None:
        framework = 'none'

    global DEFAULT_FRAMEWORK
    DEFAULT_FRAMEWORK = framework


def get_default_framework() -> framework_type:
    """Retrieve default deep learning framework set either with env variables or manually."""
    return DEFAULT_FRAMEWORK


def shortcut_module(current: str, target: str, package: Optional[str] = None) -> None:
    """Make ``current`` module an alias of ``target`` module in ``package``."""
    # Reference: https://github.com/dmlc/dgl/blob/d70a362dba8d46fd9838c79d76998a5e33f22cb7/python/dgl/nn/__init__.py#L27
    mod = importlib.import_module(target, package)
    thismod = sys.modules[current]
    for api, obj in mod.__dict__.items():
        setattr(thismod, api, obj)


def shortcut_framework(current: str) -> None:
    """Make ``current`` a shortcut of ``current.framework``."""
    if get_default_framework() != 'none':
        # Throw ModuleNotFoundError if framework is not supported
        shortcut_module(current, '.' + get_default_framework(), current)
