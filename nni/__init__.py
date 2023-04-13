# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

try:
    from .version import __version__  # pylint: disable=import-error
except ModuleNotFoundError:
    __version__ = '999.dev0'

from .runtime.log import _init_logger
_init_logger()

from .common.framework import *
from .common.serializer import trace, dump, load
from .experiment import Experiment
from .mutable.shortcut import *
from .runtime.env_vars import dispatcher_env_vars
from .runtime.log import enable_global_logging, silence_stdout
from .utils import ClassArgsValidator

if dispatcher_env_vars.SDK_PROCESS != 'dispatcher':
    from .trial import *
    from .common.nas_utils import training_update

class NoMoreTrialError(Exception):
    def __init__(self, ErrorInfo='Search space fully explored'):
        super().__init__(self)
        self.errorinfo = ErrorInfo

    def __str__(self):
        return self.errorinfo
