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
    apply_comp = ApplyCompression(model, masks_file)
    apply_comp.compress()

class ApplyCompression(Pruner):
    """
    This class is not to generate masks, but applying existing masks
    """

    def __init__(self, model, masks_file):
        """
        Parameters
        ----------
        model : torch.nn.module
            Model to be masked
        masks_file : str
            The path of user provided mask file
        """
        self.bound_model = model
        self.masks = torch.load(masks_file)
        for module_name in self.masks:
            print('module_name: ', module_name)
        config_list = self._build_config()
        super().__init__(model, config_list)

    def _build_config(self):
        op_names = []
        for module_name in self.masks:
            op_names.append(module_name)
        return [{'sparsity': 1, 'op_types': ['default', 'BatchNorm2d'], 'op_names': op_names}]

    def calc_mask(self, layer, config, **kwargs):
        """
        Directly return the corresponding mask

        Parameters
        ----------
        layer : LayerInfo
            The layer to be pruned
        config : dict
            Pruning configurations for this weight
        kwargs : dict
            Auxiliary information

        Returns
        -------
        dict
            Mask of the layer
        """
        assert layer.name in self.masks
        return self.masks[layer.name]
