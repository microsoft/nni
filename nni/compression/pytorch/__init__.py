# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .speedup import ModelSpeedup
from .compressor import Compressor, Pruner, Quantizer
from .utils.apply_compression import apply_compression_results
