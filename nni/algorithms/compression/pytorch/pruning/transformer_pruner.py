# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from schema import And, Optional

from nni.common.graph_utils import TorchModuleGraph
from nni.compression.pytorch.utils.shape_dependency import AttentionWeightDependency
from nni.compression.pytorch.utils.config_validation import CompressorSchema
from nni.compression.pytorch.compressor import Pruner
from . import L1WeightHeadMasker, L2WeightHeadMasker, L1ActivationHeadMasker, L2ActivationHeadMasker, TaylorFOHeadMasker

__all__ = ['TransformerHeadPruner']

MASKER_DICT = {
    'l1_weight': L1WeightHeadMasker,
    'l2_weight': L2WeightHeadMasker,
    'l1_activation': L1ActivationHeadMasker,
    'l2_activation': L2ActivationHeadMasker,
    'taylorfo': TaylorFOHeadMasker
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TransformerHeadPruner(Pruner):
    """
    A pruner specialized for pruning attention heads in models belong to the transformer family.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned. Expect a model from transformers library (e.g., BertModel).
        This pruner can work with other customized transformer models, but some ranking modes might fail.
    config_list : list
        Supported keys:
            - sparsity : This is to specify the sparsity operations to be compressed to.
            - op_types : Optional. Operation types to prune. (Should be 'Linear' for this pruner.)
            - op_names : Optional. Operation names to prune.
    ranking_criteria : str
        Supported criteria:
            - 'taylorfo'
            - 'l1_weight'
            - 'l2_weight'
            - 'l1_activation'
            - 'l2_activation'
    """
    def __init__(self, model, config_list, attention_name_groups=None, dummy_input=None, head_hidden_dim=None,
                 ranking_criteria='taylorfo', global_sort=False, num_iterations=1, epochs_per_iteration=1,
                 optimizer=None, trainer=None, criterion=None,
                 **algo_kwargs):
        super().__init__(model, config_list)

        self.attention_name_groups = attention_name_groups
        self.dummy_input = dummy_input
        self.head_hidden_dim = head_hidden_dim
        self.ranking_criteria = ranking_criteria
        assert self.ranking_criteria in ['l1_weight', 'l2_weight', 'l1_activation', 'l2_activation', 'taylorfo'], \
            "Unsupported ranking criteria."
        self.global_sort = global_sort
        self.num_iterations = num_iterations
        self.epochs_per_iteration = epochs_per_iteration
        self._optimizer = optimizer
        self._trainer = trainer
        self._criterion = criterion
        if self.ranking_criteria in ['l1_activation', 'l2_activation', 'taylorfo'] or num_iterations > 1:
            assert self._trainer is not None
            assert self._optimizer is not None

        # Group generation: one group per attention layer, four weights per group
        self.masking_groups = []
        if self.attention_name_groups is not None:
            logger.info("Note: weights for the same attention layer are grouped using the given attention_name_groups.")
            self.group_weights_by_name()
        else:
            assert self.dummy_input is not None
            logger.info("Note: weights for the same attention layer are grouped using model graph.")
            self._unwrap_model()
            self.group_weights_by_graph()
            self._wrap_model()

        # Group sanity check
        self.validate_weight_groups()

        # Remove any mistakenly captured ungrouped modules
        self.remove_ungrouped_modules()

        self.set_wrappers_attribute("mask_calculated", False)
        self.masker = MASKER_DICT[ranking_criteria](model, self, self.head_hidden_dim, **algo_kwargs)
        self.pruned_heads = {i: set() for i in range(len(self.masking_groups))}

    def group_weights_by_name(self):
        """
        Populate self.masking_groups using the groups specified by user in attention_name_groups.
        """
        assert len(self.masking_groups) == 0
        # build up masking groups
        name2group = {}
        for layer_idx, layer in enumerate(self.attention_name_groups):
            errmsg = 'each name group must contain 4 weights in the following order: query projection, key ' \
                     'projection, value projection, and fully connected output layer'
            assert len(layer) == 4, errmsg
            self.masking_groups.append([])
            for weight in layer:
                name2group[weight] = layer_idx
        # assign wrappers to these groups
        for wrapper in self.get_modules_wrapper():
            if wrapper.name in name2group:
                wrapper.group_idx = name2group[wrapper.name]
                self.masking_groups[name2group[wrapper.name]].append(wrapper)

        print('grouping updated:', [[x.name for x in group] for group in self.masking_groups])

    def group_weights_by_graph(self):
        """
        Populate self.masking_groups bu running inference on the module graph.
        """
        try:
            module_graph = TorchModuleGraph(self.bound_model, self.dummy_input)
            dependency_tracer = AttentionWeightDependency(traced_model=module_graph.trace)
            self.attention_name_groups = dependency_tracer.dependency_sets
            self.group_weights_by_name()

        except Exception as e:
            raise RuntimeError('Graph trace failed: please check dummy_input, or specify attention_name_groups.\n'
                               'Exception message: ' + str(e))

    def validate_weight_groups(self):
        """
        Sanity checks:
            - Q, K, V projection weights in each groups must have the same shape
            - output projection weight shape must match total hidden dimension (inferred from Q, K, V projection)
            - Four weights in a group must have the same sparsity in their config
            - If global_sort is specified, all weights must have the same sparsity
            - head_hidden_dim must be a divisor of the output dimension of the projection weights
        """
        errmsg = 'Attention weight group sanity check not passed'
        sparsity = None
        for group in self.masking_groups:
            assert len(group) == 4, errmsg + ': each group must have four weights'
            assert group[0].module.weight.size() == group[1].module.weight.size() and \
                group[1].module.weight.size() == group[2].module.weight.size(), \
                errmsg + ': the dimensions of Q, K, V projection matrices must be the same '
            assert group[0].module.weight.size()[0] == group[3].module.weight.size()[1], \
                errmsg + ': the dimension of attention results must match with input for output projection'
            assert group[0].config['sparsity'] == group[1].config['sparsity'] == \
                   group[2].config['sparsity'] == group[3].config['sparsity'], \
                errmsg + ': the sparsity of matrices in the same layer must be the same'
            if sparsity is None:
                sparsity = group[0].config['sparsity']
            if self.global_sort:
                assert sparsity == group[0].config['sparsity'], \
                    errmsg + ': for global_sort=True, the sparsity for all modules must be the same'
            t = group[0].module.weight.size(0) / self.head_hidden_dim
            assert t % 1 == 0, errmsg + ': head_hidden_dim must be a divisor of the output dimension of the ' \
                                        'projection weights'

    def remove_ungrouped_modules(self):
        """
        Remove non-attention weights that might be captured mistakenly by a simplified config_list.
        """
        care_of_modules = set([x for layer in self.masking_groups for x in layer])
        self.modules_wrapper = [x for x in self.modules_wrapper if x in care_of_modules]

    def validate_config(self, model, config_list):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Model to be pruned
        config_list : list
            List on pruning configs
        """
        schema = CompressorSchema([{
            'sparsity': And(float, lambda n: 0 < n < 1),
            Optional('op_types'): [str],
            Optional('op_names'): [str]
        }], model, logger)

        schema.validate(config_list)

    def compress(self):
        for pruning_iter in range(self.num_iterations):
            self.set_wrappers_attribute("mask_calculated", False)
            if self.ranking_criteria in ['l1_activation', 'l2_activation', 'taylorfo']:
                training = self.bound_model.training
                self.bound_model.eval()
                self._trainer(self.bound_model, optimizer=self._optimizer, criterion=self._criterion, epoch=0)
                self.update_mask()
                self.bound_model.train(training)
            else:
                self.update_mask()

            # for iterative pruning
            # finetune before next iteration
            if self.num_iterations > 1:
                for e in range(self.epochs_per_iteration):
                    self._trainer(self.bound_model, optimizer=self._optimizer, criterion=self._criterion, epoch=e+1)

            # if not the last iteration, reset the maskers (may create additional hooks)
            if self.num_iterations > 1 and pruning_iter != self.num_iterations - 1:
                self.masker.reset()

            print('pruned heads after iteration', pruning_iter, self.pruned_heads)

    def update_mask(self):
        masks_for_all_groups = None
        if self.global_sort:
            masks_for_all_groups = self._calc_mask_global()
            assert len(masks_for_all_groups) == len(self.masking_groups)
        for group_idx, layer_weight_group in enumerate(self.masking_groups):
            if self.global_sort:
                masks = masks_for_all_groups[group_idx]
            else:
                masks = self._calc_mask(layer_weight_group[0], layer_weight_group)
            if masks is not None:
                for i, mask in enumerate(masks):
                    for mask_type in mask:
                        assert hasattr(layer_weight_group[i], mask_type), \
                            "there is no attribute '%s' in wrapper on %s" % (mask_type, layer_weight_group[i])
                        setattr(layer_weight_group[i], mask_type, mask[mask_type])
                        print(f'mask updated: {layer_weight_group[i].name} {mask_type}')

    def _calc_mask(self, wrapper, weight_group, wrapper_idx=None):
        if not wrapper.mask_calculated:
            iter_sparsity = wrapper.config['sparsity'] / self.num_iterations
            masks = self.masker.calc_mask(sparsity=iter_sparsity, wrapper=wrapper, weight_group=weight_group,
                                          wrapper_idx=wrapper_idx)
            # masker.calc_mask returns None means calc_mask is not calculated successfully; can try later
            if masks is not None:
                wrapper.mask_calculated = True
            return masks
        else:
            return None

    def _calc_mask_global(self):
        if len(self.get_modules_wrapper()) == 0:
            return []
        overall_sparsity = self.get_modules_wrapper()[0].config['sparsity'] / self.num_iterations
        n_heads_total = 0
        for q_proj, _, _, _ in self.masking_groups:
            n_heads_total += int(q_proj.module.weight.size()[0] / self.head_hidden_dim)
        n_heads_to_prune = int(n_heads_total * overall_sparsity)
        return self.masker.calc_mask_global(n_heads_to_prune)

    def calc_mask(self, wrapper, **kwargs):
        raise RuntimeError("Applications should directly call TransformerHeadPruner's update_mask() method.")
