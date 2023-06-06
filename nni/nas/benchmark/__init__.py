# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

try:
    import peewee
except ImportError:
    import warnings
    warnings.warn('peewee is not installed. Please install it to use NAS benchmarks.')

del peewee

from .evaluator import *
from .space import *
from .utils import load_benchmark, download_benchmark
