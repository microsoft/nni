# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .speedup import ModelSpeedup, CalibrateType, TensorRTModelSpeedUp
from .compressor import Compressor, Pruner, Quantizer
from .pruning import apply_compression_results
