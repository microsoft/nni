# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import warnings
try:
    import peewee
except ImportError:
    warnings.warn('peewee is not installed. Please install it to use NAS benchmarks.')

from .evaluator import *
from .space import *
from .utils import load_benchmark, download_benchmark
