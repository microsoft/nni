# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import torch
import numpy as np
from .shape_dependency import ChannelDependency
# logging.basicConfig(level = logging.DEBUG)
_logger = logging.getLogger('FixMaskConflict')

class MaskConflict:
    def __init__(self, mask_file, model=None, dummy_input=None, graph=None):
        """
        MaskConflict fix the mask conflict between the layers that
        has channel dependecy with each other.

        Parameters
        ----------
        model : torch.nn.Module
            model to fix the mask conflict
        dummy_input : torch.Tensor
            input example to trace the model
        mask_file : str
            the path of the original mask file
        graph : torch._C.Graph
            the traced graph of the target model, is this parameter is not None,
            we donnot use the model and dummpy_input to get the trace graph.
        """
        # check if the parameters are valid
        parameter_valid = False
        if graph is not None:
            parameter_valid = True
        elif (model is not None) and (dummy_input is not None):
            parameter_valid = True
        if not parameter_valid:
            raise Exception('The input parameters is invalid!')
        self.model = model
        self.dummy_input = dummy_input
        self.graph = graph
        self.mask_file = mask_file
        self.masks = torch.load(self.mask_file)

    def fix_mask_conflict(self):
        """
        Fix the mask conflict before the mask inference for the layers that
        has shape dependencies. This function should be called before the
        mask inference of the 'speedup' module.
        """
        channel_depen = ChannelDependency(self.model, self.dummy_input, self.graph)
        depen_sets = channel_depen.dependency_sets
        for dset in depen_sets:
            if len(dset) == 1:
                # This layer has no channel dependency with other layers
                continue
            channel_remain = set()
            fine_grained = False
            for name in dset:
                if name not in self.masks:
                    # this layer is not pruned
                    continue
                w_mask = self.masks[name]['weight']
                shape = w_mask.size()
                count = np.prod(shape[1:])
                all_ones = (w_mask.flatten(1).sum(-1)==count).nonzero().squeeze(1).tolist()
                all_zeros = (w_mask.flatten(1).sum(-1)==0).nonzero().squeeze(1).tolist()
                if len(all_ones) + len(all_zeros) < w_mask.size(0):
                    # In fine-grained pruning, there is no need to check
                    # the shape conflict
                    _logger.info('Layers %s using fine-grained pruning', ','.join(dset))
                    fine_grained = True
                    break
                channel_remain.update(all_ones)
                _logger.debug('Layer: %s ', name)
                _logger.debug('Original pruned filters: %s', str(all_zeros))
            # Update the masks for the layers in the dependency set
            if fine_grained:
                continue
            ori_channels = 0
            for name in dset:
                mask = self.masks[name]
                w_shape = mask['weight'].size()
                ori_channels = w_shape[0]
                for i in channel_remain:
                    mask['weight'][i] = torch.ones(w_shape[1:])
                    if hasattr(mask, 'bias'):
                        mask['bias'][i] = 1
            _logger.info(','.join(dset))
            _logger.info('Pruned Filters after fixing conflict:')
            pruned_filters = set(list(range(ori_channels)))-channel_remain
            _logger.info(str(sorted(pruned_filters)))
        return self.masks

    def export(self, path):
        """
        Export the masks after fixing the conflict to file.
        """
        torch.save(self.masks, path)
