# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import torch
from .compressor import Pruner

logger = logging.getLogger('torch apply compression')

def apply_compression_results(model, masks_file):
    """
    """
    apply_comp = ApplyCompression(model, masks_file)
    apply_comp.compress()

class ApplyCompression(Pruner):
    """
    Prune to an exact pruning level specification
    """

    def __init__(self, model, masks_file):
        """
        Parameters
        ----------
        model : torch.nn.module
            Model to be pruned
        config_list : list
            List on pruning configs
        """
        self.bound_model = model
        self.masks = torch.load(masks_file)
        #ori_masks = torch.load(masks_file)
        #self.masks = {'feature.1': ori_masks['feature.1']}
        for module_name in self.masks:
            print('module_name: ', module_name)
        config_list = self._build_config()
        super().__init__(model, config_list)

    def _build_config(self):
        op_names = []
        for module_name in self.masks:
            op_names.append(module_name)
        return [{'sparsity': 1, 'op_types': ['default', 'BatchNorm2d'], 'op_names': op_names}]

    def calc_mask(self, layer, config):
        """
        """
        assert layer.name in self.masks
        #print('calc_mask: ', layer.name, self.masks[layer.name])
        print('calc_mask: ', layer.name, layer.type)
        return self.masks[layer.name]
