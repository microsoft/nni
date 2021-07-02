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
    'taylor': TaylorFOHeadMasker
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
            - 'taylor'
            - 'l1_weight'
            - 'l2_weight'
            - 'l1_activation'
            - 'l2_activation'
    """

    def __init__(self, model, config_list, attention_name_groups=None, ranking_criteria='taylor', dummy_input=None,
                 optimizer=None, trainer=None, criterion=None,
                 **algo_kwargs):
        super().__init__(model, config_list)

        self.attention_name_groups = attention_name_groups
        self.ranking_criteria = ranking_criteria
        self.dummy_input = dummy_input
        self._optimizer = optimizer
        self._trainer = trainer
        self._criterion = criterion

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
        self.masker = MASKER_DICT[ranking_criteria](model, self, **algo_kwargs)

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
            '''
            
            stack = [(name, module) for name, module in self.bound_model.named_children()]
    
            while stack:
                cur_name, cur_module = stack.pop()
                try:
                    module_graph = TorchModuleGraph(cur_module, self.dummy_input)
                    dependency_tracer = AttentionWeightDependency(traced_model=module_graph.trace)
                    weight_names_grouped.extend([[cur_name + '.' + x for x in group]
                                                 for group in dependency_tracer.dependency_sets])
                except:
                    stack.extend([(cur_name + '.' + name, module) for name, module in cur_module.named_children()])
            '''
        except Exception as e:
            raise RuntimeError('Graph trace failed: please check dummy_input, or specify attention_name_groups. '
                               'Exception message: ' + str(e))

    # TODO: more sanity checks - include head_hidden_dim parameter? sparsity agreement?
    def validate_weight_groups(self):
        errmsg = 'Attention weight group sanity check not passed'
        try:
            for group in self.masking_groups:
                assert len(group) == 4, errmsg + ': each group must have four weights'
                assert group[0].module.weight.size() == group[1].module.weight.size() and \
                    group[1].module.weight.size() == group[2].module.weight.size(), \
                    errmsg + ': the dimensions of Q, K, V projection matrices must be the same '
                assert group[0].module.weight.size()[0] == group[3].module.weight.size()[1], \
                    errmsg + ': the dimension of attention results must match with input for output projection'
        except:
            raise RuntimeError(errmsg)

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
        if self.ranking_criteria in ['l1_activation', 'l2_activation']:
            training = self.bound_model.training
            self.bound_model.eval()
            self._trainer(self.bound_model, optimizer=self._optimizer, criterion=self._criterion, epoch=0)
            self.update_mask()
            self.bound_model.train(training)
        elif self.ranking_criteria == 'taylor':
            pass
        self.update_mask()
        return self.bound_model

    def update_mask(self):
        for layer_weight_group in self.masking_groups:
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
            sparsity = wrapper.config['sparsity']
            masks = self.masker.calc_mask(sparsity=sparsity, wrapper=wrapper, weight_group=weight_group,
                                          wrapper_idx=wrapper_idx)
            # masker.calc_mask returns None means calc_mask is not calculated successfully; can try later
            if masks is not None:
                wrapper.mask_calculated = True
            return masks
        else:
            return None

    def calc_mask(self, wrapper, **kwargs):
        raise RuntimeError("Applications should directly call TransformerHeadPruner's update_mask() method.")
