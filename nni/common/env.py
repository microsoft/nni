# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from nni.typehint import Literal

framework_type = Literal['pytorch', 'tensorflow', 'mxnet', 'none']
"""Supported framework types."""

# Override these environment vars to move your cache.
ENV_NNI_HOME = 'NNI_HOME'
ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
DEFAULT_CACHE_DIR = '~/.cache'


def nni_cache_home() -> str:
    return os.path.expanduser(
        os.getenv(ENV_NNI_HOME,
                  os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'nni')))


ENV_NNI_FRAMEWORK = 'NNI_FRAMEWORK'

def framework_from_env() -> framework_type:
    framework = os.getenv(ENV_NNI_FRAMEWORK, 'pytorch')
    if framework not in framework_type.__args__:
        raise ValueError(f'{framework} does not belong to {framework_type.__args__}')
    return framework


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
    if framework_type is None:
        framework_type = 'none'

    global DEFAULT_FRAMEWORK
    DEFAULT_FRAMEWORK = framework


def get_default_framework() -> framework_type:
    """Retrieve default deep learning framework set either with env variables or manually."""
    return DEFAULT_FRAMEWORK
