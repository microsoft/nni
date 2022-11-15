# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

_logger = logging.getLogger(__name__)
warn_msg = 'import path `nni.algorithms.compression.v2` has be removed in v3.0, ' +\
           'please import from `nni.compression`.'
_logger.warning(warn_msg)
