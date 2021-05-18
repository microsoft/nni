# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
An automated gradual pruning algorithm that prunes the smallest magnitude
weights to achieve a preset level of network sparsity.
Michael Zhu and Suyog Gupta, "To prune, or not to prune: exploring the
efficacy of pruning for model compression", 2017 NIPS Workshop on Machine
Learning of Phones and other Consumer Devices.
"""

import logging
import torch
from schema import And, Optional
from .constants import MASKER_DICT
from nni.compression.pytorch.utils.config_validation import CompressorSchema
from nni.compression.pytorch.compressor import Pruner

__all__ = ['AGPPruner']

logger = logging.getLogger('torch pruner')
