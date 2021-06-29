# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from schema import And, Optional

from nni.compression.pytorch.utils.config_validation import CompressorSchema
from nni.compression.pytorch.compressor import Pruner
from . import L1WeightHeadMasker, L2WeightHeadMasker

__all__ = ['TransformerHeadPruner']

MASKER_DICT = {
    'l1_weight': L1WeightHeadMasker,
    'l2_weight': L2WeightHeadMasker
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

    def __init__(self, model, config_list, attention_name_groups=None, ranking_criteria='taylor', **algo_kwargs):
        super().__init__(model, config_list)

        self.ranking_criteria = ranking_criteria
        self.attention_name_groups = attention_name_groups
        self.masker = MASKER_DICT[ranking_criteria](model, self, **algo_kwargs)
        self.set_wrappers_attribute("mask_calculated", False)

        # Group generation: one group per attention layer, four weights per group
        self.masking_groups = []
        if self.attention_name_groups is not None:
            logger.info("Note: weights for the same attention layer are grouped using the given attention_name_groups.")
            self.group_weights_by_name()
        else:
            logger.info("Note: weights for the same attention layer are grouped using model graph.")
            self.group_weights_by_graph()

        # Group sanity check
        self.validate_weight_groups()

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

    # TODO: graph-based group inference
    def group_weights_by_graph(self):
        """
        Populate self.masking_groups bu running inference on the module graph.
        """
        pass

    # TODO: some sanity checks - weight shape agreement (including head_hidden_dim parameter)? sparsity agreement?
    def validate_weight_groups(self):
        pass

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

    def update_mask(self):
        for layer_weight_group in self.masking_groups:
            masks = self._calc_mask(layer_weight_group[0], layer_weight_group)
            if masks is not None:
                for i, mask in enumerate(masks):
                    for mask_type in mask:
                        assert hasattr(layer_weight_group[i], mask_type), \
                            "there is no attribute '%s' in wrapper on %s" % (mask_type, layer_weight_group[i])
                        setattr(layer_weight_group[i], mask_type, mask[mask_type])
                        print(f'updated {layer_weight_group[i].name} {mask_type}')

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
