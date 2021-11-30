# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import torch
from schema import Or, Optional
from nni.compression.pytorch.utils.config_validation import QuantizerSchema
from nni.compression.pytorch.compressor import Quantizer


logger = logging.getLogger(__name__)


class NaiveQuantizer(Quantizer):
    """quantize weight to 8 bits
    """

    def __init__(self, model, config_list, optimizer=None):
        super().__init__(model, config_list, optimizer)
        self.layer_scale = {}

    def validate_config(self, model, config_list):
        schema = QuantizerSchema([{
            Optional('quant_types'): ['weight'],
            Optional('quant_bits'): Or(8, {'weight': 8}),
            Optional('op_types'): [str],
            Optional('op_names'): [str],
            Optional('exclude'): bool
        }], model, logger)

        schema.validate(config_list)

    def quantize_weight(self, wrapper, **kwargs):
        weight = wrapper.module.weight
        new_scale = weight.abs().max() / 127
        scale = max(self.layer_scale.get(wrapper.name, 0), new_scale)
        self.layer_scale[wrapper.name] = scale
        orig_type = weight.type()  # TODO: user layer
        weight = weight.div(scale).type(torch.int8).type(orig_type).mul(scale)
        wrapper.module.weight = weight
        return weight
