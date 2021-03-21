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
    masks : dict/str
        A dict object that stores the masks or the path of the mask file
    model : torch.nn.Module
        model to fix the mask conflict
    dummy_input : torch.Tensor/list of tensors/dict of tensors
        input example to trace the model
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
        training = model.training
        model.eval()
        # We need to trace the model in eval mode
        traced = torch.jit.trace(model, dummy_input)
        model.train(training)

    fix_group_mask = GroupMaskConflict(masks, model, dummy_input, traced)
    masks = fix_group_mask.fix_mask()
    fix_channel_mask = ChannelMaskConflict(masks, model, dummy_input, traced)
    masks = fix_channel_mask.fix_mask()
    return masks



class MaskFix:
    def __init__(self, masks, model=None, dummy_input=None, traced=None):
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

    def fix_mask(self):
        raise NotImplementedError

    def export(self, path):
        """
        Export the masks after fixing the conflict to file.
        """
        torch.save(self.masks, path)



class GroupMaskConflict(MaskFix):
    def __init__(self, masks, model=None, dummy_input=None, traced=None):
        """
        GroupMaskConflict fix the mask conflict between the layers that
        has group dependecy with each other.

        Parameters
        ----------
        masks : dict
            a dict object that stores the masks
        model : torch.nn.Module
            model to fix the mask conflict
        dummy_input : torch.Tensor
            input example to trace the model
        traced : torch._C.torch.jit.TopLevelTracedModule
            the traced model of the target model, is this parameter is not None,
            we donnot use the model and dummpy_input to get the trace graph.
        """
        super(GroupMaskConflict, self).__init__(masks, model, dummy_input, traced)


    def fix_mask(self):
        """
        Fix the mask conflict before the mask inference for the layers that
        has group dependencies. This function should be called before the
        mask inference of the 'speedup' module.
        """
        group_depen = GroupDependency(self.model, self.dummy_input, self.traced)
        depens = group_depen.dependency
        min_groups = group_depen.min_groups
        _logger.info(depens)
        for layername in depens:
            group_max = depens[layername]
            group_min = min_groups[layername]
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
            assert shape[0] % group_max == 0
            # Find the number of masked filter for each group (mini_masked).
            # Because we have to keep the pruned filter can still
            # be divided into the same number of groups, so we only can
            # prune mini_masked filters for each group.
            step = shape[0] / group_max
            group_masked = []
            for i in range(group_max):
                _start = step * i
                _end = step * (i+1)
                _tmp_list = list(filter(lambda x: _start <= x and x < _end, all_zeros))
                group_masked.append(_tmp_list)
            mini_masked = min([len(x) for x in group_masked])
            need_unmask = set()
            for gm in group_masked:
                for i in range(mini_masked, len(gm)):
                    # To keep the output channel number still being divisible to
                    # groups, we set the masks of following filters to be zero.
                    pos = gm[i]
                    need_unmask.add(pos)
            step = shape[0] / group_min
            for i in range(group_min):
                _start = step * i
                _end = step * (i+1)
                _tmp_list = list(filter(lambda x: _start <= x and x < _end, all_zeros))
                if len(_tmp_list) == step:
                    # if the whole group is removed, then we don't have to unmask for
                    # the filters in this group
                    for pos in _tmp_list:
                        if pos in need_unmask:
                            need_unmask.remove(pos)
            for pos in need_unmask:
                    self.masks[layername]['weight'][pos] = torch.ones(shape[1:])
                    if hasattr(self.masks[layername], 'bias'):
                        self.masks[layername]['bias'][pos] = 1
        return self.masks



class ChannelMaskConflict(MaskFix):
    def __init__(self, masks, model=None, dummy_input=None, traced=None):
        """
        ChannelMaskConflict fix the mask conflict between the layers that
        has channel dependecy with each other.

        Parameters
        ----------
        masks : dict
            a dict object that stores the masks
        model : torch.nn.Module
            model to fix the mask conflict
        dummy_input : torch.Tensor
            input example to trace the model
        graph : torch._C.torch.jit.TopLevelTracedModule
            the traced graph of the target model, is this parameter is not None,
            we donnot use the model and dummpy_input to get the trace graph.
        """
        super(ChannelMaskConflict, self).__init__(masks, model, dummy_input, traced)

    def fix_mask(self):
        """
        Fix the mask conflict before the mask inference for the layers that
        has shape dependencies. This function should be called before the
        mask inference of the 'speedup' module.
        """
        channel_depen = ChannelDependency(self.model, self.dummy_input, self.traced)
        depen_sets = channel_depen.dependency_sets
        for dset in depen_sets:
            if len(dset) == 1:
                # This layer has no channel dependency with other layers
                continue
            channel_remain = set()
            fine_grained = False
            out_channels = None
            # A flag that represents if all the layers in
            # the dependency set are pruned
            all_pruned = True
            for name in dset:
                if name not in self.masks:
                    # this layer is not pruned
                    all_pruned = False
                    continue
                w_mask = self.masks[name]['weight']
                if out_channels is None:
                    out_channels = w_mask.size(0)
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
            if fine_grained or out_channels is None:
                # if use the fine-grained pruner or all the layers in
                # this dependency set are not pruned
                continue
            if not all_pruned:
                # if some layer are not pruned at all
                # then all the layers in this dependency set
                # cannot be pruned due to the shape dependency.
                channel_remain.update(range(out_channels))
            ori_channels = 0
            for name in dset:
                if name not in self.masks:
                    # this layer is not pruned at all
                    # in this case, all_pruned is False
                    # and the other layers in the same dset
                    # will not be pruned either.
                    continue
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
