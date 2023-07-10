# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import logging
import torch
import numpy as np
from .attr import get_nested_attr
from .shape_dependency import ChannelDependency, GroupDependency, InputChannelDependency
# logging.basicConfig(level = logging.DEBUG)
_logger = logging.getLogger(__name__)


# TODO: mask conflict need refactor, the current implementation is very unfriendly to the input channel masks.
def fix_mask_conflict(masks, model, dummy_input, traced=None):
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
    assert len(masks) > 0, 'Mask tensor cannot be empty'
    # if the user uses the model and dummy_input to trace the model, we
    # should get the traced model handly, so that, we only trace the
    # model once, GroupMaskConflict and ChannelMaskConflict will reuse
    # this traced model.
    if traced is None:
        assert model is not None and dummy_input is not None
        training = model.training
        # We need to trace the model in eval mode
        model.eval()
        kw_args = {}
        if torch.__version__ >= '1.6.0':
            # only pytorch with version greater than 1.6.0 has the strict option
            kw_args['strict'] = False
        try:
            import pytorch_lightning as pl
        except ImportError:
            is_lightning_module = False
        else:
            if isinstance(model, pl.LightningModule):
                is_lightning_module = True
            else:
                is_lightning_module = False
        if is_lightning_module:
            traced = model.to_torchscript(method="trace", example_inputs=dummy_input, **kw_args)
        else:
            traced = torch.jit.trace(model, dummy_input, **kw_args)
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
    def __init__(self, masks, model, dummy_input, traced=None):
        super(GroupMaskConflict, self).__init__(
            masks, model, dummy_input, traced)

    def fix_mask(self):
        """
        Fix the mask conflict before the mask inference for the layers that
        has group dependencies. This function should be called before the
        mask inference of the 'speedup' module.
        """
        group_depen = GroupDependency(
            self.model, self.dummy_input, self.traced)
        depens = group_depen.dependency
        min_groups = group_depen.min_groups
        _logger.info(depens)
        for layername in depens:
            group_max = depens[layername]
            group_min = min_groups[layername]
            if layername not in self.masks or 'weight' not in self.masks[layername]:
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
                _end = step * (i + 1)
                _tmp_list = list(
                    filter(lambda x: _start <= x and x < _end, all_zeros))
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
                _tmp_list = list(
                    filter(lambda x: _start <= x and x < _end, all_zeros))
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

    def __init__(self, masks, model, dummy_input, traced=None):
        super(ChannelMaskConflict, self).__init__(
            masks, model, dummy_input, traced)
        self.conv_prune_dim = detect_mask_prune_dim(masks, model)
        self.channel_prune_type = detect_channel_prune_type(masks, model)
        _logger.info('Dectected conv prune dim" %d', self.conv_prune_dim)

    def fix_mask(self):
        """
        Fix the mask conflict before the mask inference for the layers that
        has shape dependencies. This function should be called before the
        mask inference of the 'speedup' module. Only structured pruning masks
        are supported.
        """
        if self.conv_prune_dim == 0:
            channel_depen = ChannelDependency(
                self.model, self.dummy_input, self.traced, self.channel_prune_type)

        else:
            channel_depen = InputChannelDependency(
                self.model, self.dummy_input, self.traced)
        depen_sets = channel_depen.dependency_sets
        sum_idx = (1, 2, 3) if self.conv_prune_dim == 0 else (0, 2, 3)

        (_tmp_name, _tmp_tensor) = list(self.masks.items())[0]
        device = list(_tmp_tensor.values())[0].device

        for dset in depen_sets:
            if len(dset) <= 1:
                continue
            # channel_masks is a list, each element is None or a vector, for example:
            # [[0, 1, 1, 0, 0], [0, 0, 1, 1, 0], None], None means no channel
            # is pruned.
            channel_masks = []
            fine_grained = False
            for name in dset:
                if name in self.masks and 'weight' in self.masks[name]:
                    m = get_nested_attr(self.model, name)
                    assert m is not None
                    mask = self.masks[name]['weight']
                    if type(m).__name__ == 'Conv2d':
                        channel_mask = (mask.abs().sum(sum_idx) != 0).int()
                        if self.conv_prune_dim == 1:
                            channel_mask = channel_mask.repeat(m.groups)
                        channel_masks.append(channel_mask)
                        if (channel_mask.sum() * (mask.numel() / mask.shape[self.conv_prune_dim])).item() != (mask > 0).sum().item():
                            fine_grained = True
                    elif type(m).__name__ == 'Linear':
                        if self.conv_prune_dim == 1:
                            channel_masks.append(
                                (mask.abs().sum(0) != 0).int())
                        else:
                            channel_masks.append(
                                (mask.abs().sum(1) != 0).int())
                    elif type(m).__name__ == 'Embedding':
                        if self.conv_prune_dim == 0:
                            channel_masks.append(
                                (mask.abs().sum(0) != 0).int())
                    elif type(m).__name__ == 'BatchNorm2d':
                        channel_masks.append(mask.int())
                    elif type(m).__name__ == 'ConvTranspose2d':
                        # convtranspose have difference memory layout, so that we need create
                        # a tmp_sum_idx for conv_transpose
                        tmp_sum_idx = (
                            0, 2, 3) if self.conv_prune_dim == 0 else (1, 2, 3)
                        channel_mask = (mask.abs().sum(tmp_sum_idx) != 0).int()
                        if self.conv_prune_dim == 0:
                            channel_mask = channel_mask.repeat(m.groups)
                        channel_masks.append(channel_mask)
                        if (channel_mask.sum() * (mask.numel() / mask.shape[1 - self.conv_prune_dim])).item() != (mask > 0).sum().item():
                            fine_grained = True
                    else:
                        raise RuntimeError(
                            f'unsupported module type: {type(m).__name__}')
                else:
                    # no mask means not pruned, equivlent to full masks
                    channel_masks.append(None)
            if fine_grained:
                _logger.info("Fine-grianed mask detected")
            if all(x is None for x in channel_masks):
                continue
            num_channels_list = [len(x)
                                 for x in channel_masks if x is not None]
            # number of channels in same set should be identical
            assert len(set(num_channels_list)) == 1
            num_channels = num_channels_list[0]

            for i, dim_mask in enumerate(channel_masks):
                if dim_mask is None:
                    channel_masks[i] = torch.ones(
                        num_channels).int().to(device)

            # merge masks with 'or'
            merged_channel_mask = channel_masks[0].clone()
            for i in range(1, len(channel_masks)):
                merged_channel_mask = (
                    (merged_channel_mask + channel_masks[i]) != 0).int()

            merged_index = torch.nonzero(merged_channel_mask, as_tuple=True)[0]

            for name in dset:
                if name not in self.masks or 'weight' not in self.masks[name]:
                    assert all(merged_channel_mask)
                    continue
                orig_mask = self.masks[name]['weight']
                m = get_nested_attr(self.model, name)
                new_mask = torch.zeros_like(orig_mask)
                if type(m).__name__ == 'Conv2d':
                    if self.conv_prune_dim == 0:
                        new_mask[merged_index, :, :, :] = 1.
                    else:
                        new_mask[:, torch.nonzero(merged_channel_mask[:new_mask.shape[1]], as_tuple=True)[0], :, :] = 1.
                elif type(m).__name__ == 'Linear':
                    if self.conv_prune_dim == 0:
                        new_mask[merged_index, :] = 1.
                    elif self.conv_prune_dim == 1:
                        new_mask[:, merged_index] = 1.
                elif type(m).__name__ == 'Embedding':
                    if self.conv_prune_dim == 0:
                        new_mask[:, merged_index] = 1.
                elif type(m).__name__ == 'BatchNorm2d':
                    new_mask = merged_channel_mask.type_as(orig_mask)
                else:
                    raise RuntimeError(
                        f'unsupported module type: {type(m).__name__}')
                self.masks[name]['weight'] = new_mask
                if 'bias' in self.masks[name] and self.masks[name]['bias'] is not None:
                    if self.conv_prune_dim == 0:
                        self.masks[name]['bias'] = merged_channel_mask.type_as(
                            self.masks[name]['bias'])

        return self.masks

def detect_channel_prune_type(masks, model):
    """
    User can prune a channel through two ways: 1) prune
    the corresponding filter of the conv layer(all the
    filter related pruner), 2) prune the BN layers that
    followed after a conv(Slim pruner). This function find
    the pruning type of the masks.

    Parameters
    ----------
    masks: dict
        A dict object that stores the masks.
    model: nn.Module
        Model object which the mask can be applied on.

    Returns:
    -------
    prune_type: str
        Could be Filter or Batchnorm
    """
    prune_type = 'Filter'
    all_batch_norm = True
    for layer_name in masks:
        m = get_nested_attr(model, layer_name)
        if m is None or (not isinstance(m, torch.nn.BatchNorm2d)):
            all_batch_norm = False
            break
    if all_batch_norm:
        # if all masks are for batchnorm layers, then the prune_type is BatchNorm
        # Note, actually we currently do not support pruning both Conv and BatchNorm
        # at the same time.
        prune_type = 'Batchnorm'
    return prune_type

def detect_mask_prune_dim(masks, model):
    """
    Detect how the masks of convolutional layers are pruned.

    Parameters
    ----------
    masks: dict
        A dict object that stores the masks.
    model: nn.Module
        Model object which the mask can be applied on.
    Returns:
    -------
        How the masks of convolutional layers are pruned, this depends on pruning algorithms, it should
        return 1 for masks generated by AMCPruner, and returns 0 for masks generated by the rest
        NNI builtin pruners.
        0: filter pruning, prune filters of weights which causes channels of output feature maps are pruned.
        1: channel pruning, prune kernels corresponding to each input channels which causes channels of
           input feature maps are pruned.
    """
    dim0_preserved, dim1_preserved = 0., 0.
    dim0_num, dim1_num = 0., 0.
    for module_name in masks:
        if 'weight' not in masks[module_name]:
            continue

        m = get_nested_attr(model, module_name)
        if m is None or type(m).__name__ != 'Conv2d':
            continue

        mask = masks[module_name]['weight'].clone()
        assert (mask >= 0).sum() == mask.numel(), \
            "mask values should be greater than or equal to 0."
        mask = (mask > 0).int()
        mask = mask.view(mask.shape[0], mask.shape[1], -1)
        dim0_mask = (mask.sum((1, 2)) > 0).int()
        dim1_mask = (mask.sum((0, 2)) > 0).int()
        dim0_preserved += dim0_mask.sum().item()
        dim1_preserved += dim1_mask.sum().item()
        dim0_num += len(dim0_mask)
        dim1_num += len(dim1_mask)

    if dim0_num == 0 or dim1_num == 0:
        _logger.warning('no multi-dimension masks found.')
        return 0

    dim0_sparsity, dim1_sparsity = 1. - dim0_preserved / \
        dim0_num, 1. - dim1_preserved / dim1_num
    _logger.info('dim0 sparsity: %f', dim0_sparsity)
    _logger.info('dim1 sparsity: %f', dim1_sparsity)

    if dim0_sparsity == dim1_sparsity == 0.:
        _logger.warning('nothing masked.')

    if dim0_sparsity > 0 and dim1_sparsity > 0:
        _logger.warning('both dim0 and dim1 masks found.')

    return 0 if dim0_sparsity >= dim1_sparsity else 1
