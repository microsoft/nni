# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from nni.experiment.config.base import ConfigBase


@dataclass
class QuantizerConfig(ConfigBase):
    """
    A placeholder for quantizer config.
    Use to config the initialization parameters of a quantizer used in the compression experiment.
    """
    pass
