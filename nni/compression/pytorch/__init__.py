# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from nni.runtime.importing import _shortcut

from .speedup import ModelSpeedup
from .compressor import Compressor, Pruner, Quantizer
from .pruning import apply_compression_results

_shortcut(__name__, [
    'nni.algorithms.compression.pytorch.pruning',
    'nni.algoirthms.compression.pytorch.quantization'
])
