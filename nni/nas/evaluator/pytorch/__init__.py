# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import warnings

try:
    from .lightning import *
except ImportError:
    warnings.warn("PyTorch-Lightning must be installed to use PyTorch in NAS. "
                  "If you are not using PyTorch, please `nni.set_default_framework('none')`")
    raise
