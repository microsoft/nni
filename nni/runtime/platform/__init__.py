# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from ..env_vars import trial_env_vars, dispatcher_env_vars

assert dispatcher_env_vars.SDK_PROCESS != 'dispatcher'

if os.environ.get('NNI_TRIAL_COMMAND_CHANNEL'):
    from .v3 import *
elif trial_env_vars.NNI_PLATFORM is None:
    from .standalone import *
elif trial_env_vars.NNI_PLATFORM == 'unittest':
    from .test import *
else:
    from .local import *
