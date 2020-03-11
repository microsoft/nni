# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import torch
from .compressor import Pruner

logger = logging.getLogger('torch apply compression')

def apply_compression_results(model, masks_file):
    """
    Apply the masks from ```masks_file``` to the model

    Parameters
    ----------
    model : torch.nn.module
        The model to be compressed
    masks_file : str
        The path of the mask file
    """
    masks = torch.load(masks_file)
    for name, module in model.named_modules():
        if name in masks:
            module.weight.data = module.weight.data.mul_(masks[name]['weight_mask'])
            if 'bias_mask' in masks[name]:
                module.bias.data = module.bias.data.mul_(masks[name]['bias_mask'])