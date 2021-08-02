# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import torch
from .weight_masker import WeightMasker

__all__ = ['L1WeightHeadMasker', 'L2WeightHeadMasker', 'L1ActivationHeadMasker', 'L2ActivationHeadMasker',
           'TaylorFOHeadMasker']

logger = logging.getLogger('transformer head pruner')


class AttentionHeadMasker(WeightMasker):
    """
    A structured pruning masker base class that prunes attention heads in attention layers.

    Parameters
    ----------
    model: nn.Module
        model to be pruned
    pruner: Pruner
        A Pruner instance used to prune the model
    head_hidden_dim: int
        Hidden dimension for each attention head (e.g., 64 for BERT base)
    """
    def __init__(self, model, pruner, head_hidden_dim=None):
        super().__init__(model, pruner)
        self.head_hidden_dim = head_hidden_dim
        assert self.head_hidden_dim is not None, "head_hidden_dim must be specified."

    def reset(self):
        """
        Derived classes can override this method to do preparations necessary for calculating importance scores.
        This method is called during iterative pruning, before each iteration starts (except the first one).
        """
        pass

    def calc_mask(self, sparsity, wrapper=None, wrapper_idx=None, weight_group=None, **kwargs):
        """
        Calculate all the masks for a group of wrappers (specified in weight_group).
        This function only utilizes local information for mask calculation. If global_sort is specified for the pruner,
        the pruner should call calc_mask_global instead of this function.

        Parameters
        ----------
        sparsity: float
            The target (amount of increase of) sparsity of the wrapper list.
        weight_group: list
            A four-element list of module wrappers
        wrapper: PrunerModuleWrapper/list of PrunerModuleWrappers
            Should be None. Not used in this masker, just for consistency with the parent API.
        wrapper_idx: int/list of int
            Should be None. Not used in this masker, just for consistency with the parent API.
        Returns
        -------
        masks : list
            masks for each element in the group.
            Each element in the list masks is a dictionary for storing masks, keys of the dict:
                'weight_mask':  weight mask tensor
                'bias_mask': bias mask tensor (optional)
        """
        assert weight_group is not None
        if len(weight_group) == 0:
            return None
        else:
            num_total = weight_group[0].module.weight.data.size(0) // self.head_hidden_dim
            if num_total < 2:
                return None
            num_prune = max(int(num_total * sparsity), 1)
            return self.get_mask(num_prune, weight_group, **kwargs)

    def calc_mask_global(self, n_heads_to_prune):
        """
        Calculate all the masks for all groups in the pruner.

        Parameters
        ----------
        n_heads_to_prune : int
            Total number of attention heads to prune.
        Returns
        -------
        all_masks : list
            A list of masks for all groups, where each element is a list of masks for each module in the group.
        """
        # calculate scores as normal (this step does not require global information)
        head_importance_scores = []
        for group_idx, group in enumerate(self.pruner.masking_groups):
            if len(group) != 0:
                scores = self.get_head_importance_scores(group)
                n_heads = group[0].module.weight.size(0) // self.head_hidden_dim
                for head_idx in range(n_heads):
                    head_importance_scores.append([group_idx, head_idx, scores[head_idx]])

        # determine which head to prune for each layer
        n_selected = 0
        for group_idx, head_idx, _ in sorted(head_importance_scores, key=(lambda x: x[-1])):
            n_heads_original = self.pruner.masking_groups[group_idx][0].module.weight.size(0) // self.head_hidden_dim
            n_heads_remaining = n_heads_original - len(self.pruner.pruned_heads[group_idx])
            if n_heads_remaining > 1 and head_idx not in self.pruner.pruned_heads[group_idx]:
                self.pruner.pruned_heads[group_idx].add(head_idx)
                n_selected += 1
            if n_selected >= n_heads_to_prune:
                break

        # generate masks
        all_masks = []
        for group_idx, group in enumerate(self.pruner.masking_groups):
            if len(group) == 0:
                masks = None
            else:
                n_heads = group[0].module.weight.size(0) // self.head_hidden_dim
                device = group[0].module.weight.device
                head_level_mask = torch.tensor([i not in self.pruner.pruned_heads[group_idx] for i in range(n_heads)], device=device)  # pylint: disable=not-callable
                masks = self._get_layer_masks_from_head_mask(group, head_level_mask)
            all_masks.append(masks)

        return all_masks

    def get_mask(self, num_prune, weight_group, **kwargs):
        """
        Calculate the mask of given layer (weight_group).

        Parameters
        ----------
        num_prune: int
            Num of heads to prune
        weight_group: list
            A four-element list of module wrappers
        Returns
        -------
        masks : list
            masks for each element in the group.
            Each element in the list masks is a dictionary for storing masks, keys of the dict:
                'weight_mask':  weight mask tensor
                'bias_mask': bias mask tensor (optional)
        """
        raise NotImplementedError('{} get_mask is not implemented'.format(self.__class__.__name__))

    def _get_layer_masks_from_head_mask(self, weight_group, head_mask_bool, device=None):
        q_proj, _, _, output_proj = weight_group
        if device is None:
            device = q_proj.module.weight.device

        n_heads = q_proj.module.weight.size()[0] // self.head_hidden_dim
        weight_mask_shape = q_proj.module.weight.data.view([n_heads, -1]).size()
        bias_mask_shape = q_proj.module.bias.data.view([n_heads, -1]).size()

        mask_weight = head_mask_bool.unsqueeze(-1).expand(weight_mask_shape).type_as(q_proj.module.weight)
        mask_bias = head_mask_bool.unsqueeze(-1).expand(bias_mask_shape).type_as(q_proj.module.weight)

        mask_weight_proj = mask_weight.contiguous().view(q_proj.module.weight.size()).detach().to(device)
        mask_bias_proj = mask_bias.contiguous().view(-1).detach().to(device)
        masks_for_proj = {'weight_mask': mask_weight_proj.detach()}
        if hasattr(q_proj.module, 'bias') and q_proj.module.bias is not None:
            masks_for_proj['bias_mask'] = mask_bias_proj

        mask_weight_dense = mask_bias_proj.expand_as(output_proj.module.weight.data).detach().to(device)
        mask_bias_dense = torch.ones_like(output_proj.module.bias.data).to(device)
        masks_for_dense = {'weight_mask': mask_weight_dense.detach()}
        if hasattr(output_proj.module, 'bias') and output_proj.module.bias is not None:
            masks_for_dense['bias_mask'] = mask_bias_dense

        masks = [masks_for_proj, masks_for_proj, masks_for_proj, masks_for_dense]

        return masks

    def get_mask_by_importance_ranking(self, num_prune, weight_group):
        """
        Calculate the mask of given layer by pruning out heads with lowest importance scores.

        Parameters
        ----------
        num_prune: int
            Num of heads to prune
        weight_group: list
            list of a group of weights for an attention layer
        Returns
        -------
        masks : list
            masks for each element in the group.
            Each element in the list masks is a dictionary for storing masks, keys of the dict:
                'weight_mask':  weight mask tensor
                'bias_mask': bias mask tensor (optional)
        """
        importance_scores = self.get_head_importance_scores(weight_group)
        if importance_scores is None:
            return None

        importance_scores = [[i, importance_scores[i]] for i in range(len(importance_scores))]
        head_mask_bool = torch.ones(len(importance_scores))
        n_selected = 0
        for head_idx, _ in sorted(importance_scores, key=(lambda x: x[-1])):
            head_mask_bool[head_idx] = 0
            if head_idx not in self.pruner.pruned_heads[weight_group[0].group_idx]:
                n_selected += 1
                # update pruned_heads in pruner (mainly for iterative pruning)
                self.pruner.pruned_heads[weight_group[0].group_idx].add(head_idx)
            if n_selected == num_prune:
                break

        return self._get_layer_masks_from_head_mask(weight_group, head_mask_bool)

    def get_head_importance_scores(self, weight_group):
        """
        Calculate the importance score for each head.
        Parameters
        ----------
        weight_group: list
            list of a group of weights for an attention layer

        Returns
        -------
        importance_scores: tensor
            Tensor that indicates the importance of each head
        """
        raise NotImplementedError('{} get_channel_sum is not implemented'.format(self.__class__.__name__))


class L1WeightHeadMasker(AttentionHeadMasker):
    """
    A structured pruning algorithm that prunes the heads weight smallest weight magnitude for the query, head,
    and key projection matrices. L1 norm is used for magnitude calculation. Note that in this implementation, weight
    norms of q_proj, k_proj, v_proj from each head are summed as the final importance score for the head.
    """
    def get_head_importance_scores(self, weight_group):
        q_proj, k_proj, v_proj, _ = weight_group

        n_heads = q_proj.module.weight.size()[0] // self.head_hidden_dim
        query_proj_weights = q_proj.module.weight.data.view([n_heads, -1])
        key_proj_weights = k_proj.module.weight.data.view([n_heads, -1])
        value_proj_weights = v_proj.module.weight.data.view([n_heads, -1])

        query_norm_avg = torch.norm(query_proj_weights, 1, -1)
        key_norm_avg = torch.norm(key_proj_weights, 1, -1)
        value_norm_avg = torch.norm(value_proj_weights, 1, -1)

        return ((query_norm_avg + key_norm_avg + value_norm_avg) / 3).detach()

    def get_mask(self, num_prune, weight_group, **kwargs):
        return self.get_mask_by_importance_ranking(num_prune, weight_group)


class L2WeightHeadMasker(AttentionHeadMasker):
    """
    A structured pruning algorithm that prunes the heads weight smallest weight magnitude for the query, head,
    and key projection matrices. L2 norm is used for magnitude calculation. Note that in this implementation, weight
    norms of q_proj, k_proj, v_proj from each head are summed as the final importance score for the head.
    """
    def get_head_importance_scores(self, weight_group):
        q_proj, k_proj, v_proj, _ = weight_group

        n_heads = q_proj.module.weight.size()[0] // self.head_hidden_dim
        query_proj_weights = q_proj.module.weight.data.view([n_heads, -1])
        key_proj_weights = k_proj.module.weight.data.view([n_heads, -1])
        value_proj_weights = v_proj.module.weight.data.view([n_heads, -1])

        query_norm_avg = torch.norm(query_proj_weights, 2, -1)
        key_norm_avg = torch.norm(key_proj_weights, 2, -1)
        value_norm_avg = torch.norm(value_proj_weights, 2, -1)

        return ((query_norm_avg + key_norm_avg + value_norm_avg) / 3).detach()

    def get_mask(self, num_prune, weight_group, **kwargs):
        return self.get_mask_by_importance_ranking(num_prune, weight_group)


class L1ActivationHeadMasker(AttentionHeadMasker):
    """
    A structured pruning algorithm that prunes the heads with smallest final output value.
    Note that this masker only relies on the output of the output layer of each attention layer.
    The masker collects the L1 norm of the output of the last weight (output projection) in each group on the entire
    train set, and prunes the heads producing the smallest output.
    """
    def __init__(self, model, pruner, head_hidden_dim=None):
        super().__init__(model, pruner, head_hidden_dim)
        self.reset()

    def reset(self):
        self.pruner.hook_id = self._add_activation_collector(self.pruner)

    def get_head_importance_scores(self, weight_group):
        _, _, _, output_proj = weight_group
        activations = torch.stack(self.pruner.collected_activation[output_proj.group_idx], -1)
        activations = torch.sum(activations, -1)
        n_heads = activations.size()[0] // self.head_hidden_dim
        scores = torch.sum(activations.view([n_heads, -1]), -1).detach().cpu()

        # clean up hooks
        if self.pruner.hook_id in self.pruner._fwd_hook_handles:
            self.pruner.remove_activation_collector(self.pruner.hook_id)

        return scores

    def _add_activation_collector(self, pruner):
        def collector(collected_activation):
            def hook(module_, input_, output):
                if type(input_) is tuple:
                    input_ = input_[0]
                raw_activation = torch.abs(input_.detach().cpu())               # L1-norm
                raw_activation_reduced = torch.sum(raw_activation, [0, 1])
                collected_activation.append(raw_activation_reduced)
            return hook
        pruner.collected_activation = {}
        pruner._fwd_hook_id += 1
        pruner._fwd_hook_handles[pruner._fwd_hook_id] = []

        for _, _, _, output_proj in pruner.masking_groups:
            pruner.collected_activation[output_proj.group_idx] = []
            handle = output_proj.register_forward_hook(collector(pruner.collected_activation[output_proj.group_idx]))

            pruner._fwd_hook_handles[pruner._fwd_hook_id].append(handle)

        return pruner._fwd_hook_id

    def get_mask(self, num_prune, weight_group, **kwargs):
        return self.get_mask_by_importance_ranking(num_prune, weight_group)


class L2ActivationHeadMasker(AttentionHeadMasker):
    """
    A structured pruning algorithm that prunes the heads with smallest final output value.
    Note that this masker only relies on the output of the output layer of each attention layer.
    The masker collects the L2 norm of the output of the last weight (output projection) in each group on the entire
    train set, and prunes the heads producing the smallest output.
    """
    def __init__(self, model, pruner, head_hidden_dim=None):
        super().__init__(model, pruner, head_hidden_dim)
        self.reset()

    def reset(self):
        self.pruner.hook_id = self._add_activation_collector(self.pruner)

    def get_head_importance_scores(self, weight_group):
        _, _, _, output_proj = weight_group
        activations = torch.stack(self.pruner.collected_activation[output_proj.group_idx], -1)
        scores = torch.sum(activations, -1).detach().cpu()
        # n_heads = activations.size()[0] // self.head_hidden_dim
        # scores = torch.sum(activations.view([n_heads, -1]), -1).detach().cpu()

        # clean up hooks
        if self.pruner.hook_id in self.pruner._fwd_hook_handles:
            self.pruner.remove_activation_collector(self.pruner.hook_id)

        return scores

    def _add_activation_collector(self, pruner):
        def collector(collected_activation, head_hidden_dim):
            def hook(module_, input_, output):
                if type(input_) is tuple:
                    input_ = input_[0]
                raw_activation = input_.detach().cpu() ** 2
                n_heads = raw_activation.size(-1) // head_hidden_dim
                raw_activation = raw_activation.view(raw_activation.size(0), raw_activation.size(1), n_heads, -1)
                raw_activation = torch.norm(raw_activation, 2, -1)           # (B, S, n_heads)
                raw_activation_reduced = torch.sum(raw_activation, [0, 1])          # (n_heads,)
                collected_activation.append(raw_activation_reduced)

            return hook

        pruner.collected_activation = {}
        pruner._fwd_hook_id += 1
        pruner._fwd_hook_handles[pruner._fwd_hook_id] = []

        for _, _, _, output_proj in pruner.masking_groups:
            pruner.collected_activation[output_proj.group_idx] = []
            handle = output_proj.register_forward_hook(collector(pruner.collected_activation[output_proj.group_idx],
                                                                 head_hidden_dim=self.head_hidden_dim))

            pruner._fwd_hook_handles[pruner._fwd_hook_id].append(handle)

        return pruner._fwd_hook_id

    def get_mask(self, num_prune, weight_group, **kwargs):
        return self.get_mask_by_importance_ranking(num_prune, weight_group)


class TaylorFOHeadMasker(AttentionHeadMasker):
    """
    A structured pruning algorithm that prunes the heads with smallest final output contribution.
    Note that this masker only relies on the output of the output layer of each attention layer.
    The masker collects the output the last weight (output projection) in each group and the corresponding gradient
    on the entire train set, and prunes the heads producing the smallest contribution as used in the following papers:
        "Are Sixteen Heads Really Better than One?" (Michel et.al, 2019)
        "Pruning convolutional neural networks for resource efficient inference." (Molchanov et. al., 2017)
    """
    def __init__(self, model, pruner, head_hidden_dim=None):
        super().__init__(model, pruner, head_hidden_dim)
        self.reset()

    def reset(self):
        self.pruner.hook_id = self._add_activation_collector()  # forward hooks for collecting activation
        self.backward_hooks = {}  # backward hooks for collecting gradient
        self._add_gradient_collector()

    def get_head_importance_scores(self, weight_group):
        _, _, _, output_proj = weight_group
        result = output_proj.head_importance_scores

        # clean up hooks and cached data
        if self.pruner.hook_id in self.pruner._fwd_hook_handles:
            self.pruner.remove_activation_collector(self.pruner.hook_id)
        self.backward_hooks[output_proj.group_idx].remove()
        for attr in ['forward_output_cached', 'head_importance_scores']:
            output_proj.__dict__.pop(attr, None)

        return result

    def _add_activation_collector(self):
        def forward_hook(md, inp, out):
            if type(inp) is tuple:
                inp = inp[0]
            n_heads_per_layer = inp.size(-1) // self.head_hidden_dim
            heads_output = inp.view([inp.size(0), inp.size(1), n_heads_per_layer, -1]).detach()
            md.forward_output_cached = heads_output

        self.pruner._fwd_hook_id += 1
        self.pruner._fwd_hook_handles[self.pruner._fwd_hook_id] = []

        for _, _, _, output_proj in self.pruner.masking_groups:
            handle = output_proj.register_forward_hook(forward_hook)
            self.pruner._fwd_hook_handles[self.pruner._fwd_hook_id].append(handle)

        return self.pruner._fwd_hook_id

    def _add_gradient_collector(self):
        def grad_hook(md, grad_in, grad_out):
            if type(grad_in) is tuple:
                grad_in = grad_in[0]
            n_heads_per_layer = grad_in.size(-1) // self.head_hidden_dim
            heads_grad = grad_in.view([grad_in.size(0), grad_in.size(1), n_heads_per_layer, -1])
            heads_scores = torch.abs(heads_grad * md.forward_output_cached)
            heads_scores = torch.sum(heads_scores, [0, 1, 3]).detach().cpu()
            if hasattr(md, 'head_importance_scores'):
                md.head_importance_scores += heads_scores
            else:
                md.head_importance_scores = heads_scores

        for _, _, _, output_proj in self.pruner.masking_groups:
            handle = output_proj.register_backward_hook(grad_hook)
            self.backward_hooks[output_proj.group_idx] = handle

    def get_mask(self, num_prune, weight_group, **kwargs):
        return self.get_mask_by_importance_ranking(num_prune, weight_group)
