# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import logging
import torch
import numpy as np
from .shape_dependency import ChannelDependency, GroupDependency
# logging.basicConfig(level = logging.DEBUG)
_logger = logging.getLogger('FixMaskConflict')

def fix_mask_conflict(masks, model=None, dummy_input=None, traced=None):
    """
    MaskConflict fix the mask conflict for the channel dependencies
    and group dependency.

    Parameters
    ----------
    model : torch.nn.Module
        model to fix the mask conflict
    dummy_input : torch.Tensor
        input example to trace the model
    masks : dict/str
        a dict object that stores the masks or the path of the mask file
    traced : torch._C.torch.jit.TopLevelTracedModule
        the traced model of the target model, is this parameter is not None,
        we donnot use the model and dummpy_input to get the trace graph.
    """
    if isinstance(masks, str):
        # if the input is the path of the mask_file
        assert os.path.exists(masks)
        masks = torch.load(masks)
    # if the user uses the model and dummy_input to trace the model, we
    # should get the traced model handly, so that, we only trace the
    # model once, GroupMaskConflict and ChannelMaskConflict will reuse
    # this traced model.
    if traced is None:
        assert model is not None and dummy_input is not None
        with torch.onnx.set_training(model, False):
            # We need to trace the model in this way, else it will have problems
            traced = torch.jit.trace(model, dummy_input)

    fix_group_mask = GroupMaskConflict(masks, model, dummy_input, traced)
    masks = fix_group_mask.fix_mask_conflict()
    fix_channel_mask = ChannelMaskConflict(masks, model, dummy_input, traced)
    masks = fix_channel_mask.fix_mask_conflict()
    return masks


class GroupMaskConflict:
    def __init__(self, masks, model=None, dummy_input=None, traced=None):
        """
        GroupMaskConflict fix the mask conflict between the layers that
        has group dependecy with each other.

        Parameters
        ----------
        model : torch.nn.Module
            model to fix the mask conflict
        dummy_input : torch.Tensor
            input example to trace the model
        masks : dict
            a dict object that stores the masks
        traced : torch._C.torch.jit.TopLevelTracedModule
            the traced model of the target model, is this parameter is not None,
            we donnot use the model and dummpy_input to get the trace graph.
        """
        # check if the parameters are valid
        parameter_valid = False
        if traced is not None:
            parameter_valid = True
        elif (model is not None) and (dummy_input is not None):
            parameter_valid = True
        if not parameter_valid:
            raise Exception('The input parameters is invalid!')
        self.model = model
        self.dummy_input = dummy_input
        self.traced = traced
        self.masks = masks

    def fix_mask_conflict(self):
        """
        Fix the mask conflict before the mask inference for the layers that
        has group dependencies. This function should be called before the
        mask inference of the 'speedup' module.
        """
        group_depen = GroupDependency(self.model, self.dummy_input, self.traced)
        depens = group_depen.dependency
        _logger.info(depens)
        for layername in depens:
            group = depens[layername]
            if layername not in self.masks:
                # this layer not pruned
                continue
            w_mask = self.masks[layername]['weight']
            shape = w_mask.size()
            count = np.prod(shape[1:])
            all_ones = (w_mask.flatten(1).sum(-1) == count).nonzero().squeeze(1).tolist()
            all_zeros = (w_mask.flatten(1).sum(-1) == 0).nonzero().squeeze(1).tolist()
            if len(all_ones) + len(all_zeros) < w_mask.size(0):
                # In fine-grained pruning, skip this layer
                _logger.info('Layers %s using fine-grained pruning', layername)
                continue
            assert shape[0] % group == 0
            # Find the number of masked filter for each group (mini_masked).
            # Because we have to keep the pruned filter can still
            # be divided into the same number of groups, so we only can
            # prune mini_masked filters for each group.
            step = shape[0] / group
            group_masked = []
            for i in range(group):
                _start = step * i
                _end = step * (i+1)
                _tmp_list = list(filter(lambda x: _start <= x and x < _end, all_zeros))
                group_masked.append(_tmp_list)
            mini_masked = min([len(x) for x in group_masked])
            for gm in group_masked:
                for i in range(mini_masked, len(gm)):
                    # To keep the output channel number still being divisible to
                    # groups, we set the masks of following filters to be zero.
                    pos = gm[i]
                    self.masks[layername]['weight'][pos] = torch.ones(shape[1:])
                    if hasattr(self.masks[layername], 'bias'):
                        self.masks[layername]['bias'][pos] = 1
        return self.masks

    def export(self, path):
        """
        Export the masks after fixing the conflict to file.
        """
        torch.save(self.masks, path)


class ChannelMaskConflict:
    def __init__(self, masks, model=None, dummy_input=None, graph=None):
        """
        ChannelMaskConflict fix the mask conflict between the layers that
        has channel dependecy with each other.

        Parameters
        ----------
        model : torch.nn.Module
            model to fix the mask conflict
        dummy_input : torch.Tensor
            input example to trace the model
        masks : dict
            a dict object that stores the masks
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
        self.masks = masks

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
                all_ones = (w_mask.flatten(1).sum(-1) == count).nonzero().squeeze(1).tolist()
                all_zeros = (w_mask.flatten(1).sum(-1) == 0).nonzero().squeeze(1).tolist()
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
