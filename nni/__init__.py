# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .version import __version__

from .runtime.log import init_logger
init_logger()

from .runtime.env_vars import dispatcher_env_vars
from .utils import ClassArgsValidator

if dispatcher_env_vars.SDK_PROCESS != 'dispatcher':
    from .trial import *
    from .smartparam import *
    from .common.nas_utils import training_update

class NoMoreTrialError(Exception):
    def __init__(self, ErrorInfo):
        super().__init__(self)
        self.errorinfo = ErrorInfo

    def __str__(self):
        return self.errorinfo
