# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import torch

logger = logging.getLogger('torch apply compression')

def apply_compression_results(model, masks_file, map_location=None):
    """
    Apply the masks from ```masks_file``` to the model
    Note: this API is for inference, because it simply multiplies weights with
    corresponding masks when this API is called.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be compressed
    masks_file : str
        The path of the mask file
    map_location : str
        the device on which masks are placed, same to map_location in ```torch.load```
    """
    masks = torch.load(masks_file, map_location)
    for name, module in model.named_modules():
        if name in masks:
            module.weight.data = module.weight.data.mul_(masks[name]['weight'])
            if hasattr(module, 'bias') and module.bias is not None and 'bias' in masks[name]:
                module.bias.data = module.bias.data.mul_(masks[name]['bias'])