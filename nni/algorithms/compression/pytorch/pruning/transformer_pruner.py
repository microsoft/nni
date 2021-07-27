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
    head_hidden_dim : int
        Dimension of the hidden dimension of each attention head. (e.g., 64 for BERT)
        We assume that this head_hidden_dim is constant across the entire model.
    attention_name_groups : list (Optional)
        List of groups of names for weights of each attention layer. Each element should be a four-element list, with
        the first three corresponding to Q_proj, K_proj, V_proj (in any order) and the last one being output_proj.
    dummy_input : torch.Tensor (Optional)
        Input to model's forward method, used to infer module grouping if attention_name_groups is not specified.
        This tensor is used by the underlying torch.jit.trace to infer the module graph.
    ranking_criterion : str
        The criterion for ranking attention heads. Currently we support:
            - l1_weight: l1 norm of Q_proj, K_proj, and V_proj
            - l2_weight: l2 norm of Q_proj, K_proj, and V_proj
            - l1_activation: l1 norm of the output of attention computation
            - l2_activation: l2 norm of the output of attention computation
            - taylorfo: l1 norm of the output of attention computation * gradient for this output
                        (check more details in the masker documentation)
    global_sort : bool
        Whether rank the heads globally or locally before deciding heads to prune.
    num_iterations : int
        Number of pruning iterations. Defaults to 1 (ont-shot pruning). If num_iterations > 1, the pruner will split
        the sparsity specified in config_list uniformly and assign a fraction to each pruning iteration.
    epochs_per_iteration : int
        Number of finetuning epochs before the next pruning iteration.
        Only used when num_iterations > 1.
        If num_iterations is 1, then no finetuning is performed by the pruner after pruning.
    optimizer: torch.optim.Optimizer
        Optimizer used to train model
    trainer: function
        Function used to finetune the model between pruning iterations.
        Only used when  num_iterations > 1 or ranking_criterion is 'taylorfo'.
        Users should write this function as a normal function to train the PyTorch model and include
        `model, optimizer, criterion, epoch` as function arguments. Note that the trainer is also used for collecting
        gradients for pruning if ranking_criterion is 'taylorfo'. In that case, ``epoch=None`` will be passed.
    criterion: function
        Function used to calculate the loss between the target and the output.
        Only used when  num_iterations > 1 or ranking_criterion is 'taylorfo'.
        For example, you can use ``torch.nn.CrossEntropyLoss()`` as input.
    forward_runner: function
        Function used to perform a "dry run" on the model on the entire train/validation dataset in order to collect
        data for pruning required by the criteria 'l1_activation' or 'l2_activation'.
        Only used when ranking_criterion is 'l1_activation' or 'l2_activation'.
        Users should write this function as a normal function that accepts a PyTorch model and runs forward on the model
        using the entire train/validation dataset. This function is not expected to perform any backpropagation or
        parameter updates.
    """
    def __init__(self, model, config_list,head_hidden_dim, attention_name_groups=None, dummy_input=None,
                 ranking_criterion='l1_weight', global_sort=False, num_iterations=1, epochs_per_iteration=1,
                 optimizer=None, trainer=None, criterion=None, forward_runner=None,
                 **algo_kwargs):
        super().__init__(model, config_list)

        self.head_hidden_dim = int(head_hidden_dim)
        self.attention_name_groups = attention_name_groups
        self.dummy_input = dummy_input
        self.ranking_criterion = ranking_criterion
        assert self.ranking_criterion in ['l1_weight', 'l2_weight', 'l1_activation', 'l2_activation', 'taylorfo'], \
            "Unsupported ranking criteria."
        self.global_sort = global_sort
        self.num_iterations = int(num_iterations)
        assert self.num_iterations >= 1, "num_iterations must be greater than or equal to 1"
        self.epochs_per_iteration = int(epochs_per_iteration)
        self._optimizer = optimizer
        self._trainer = trainer
        self._criterion = criterion
        self._forward_runner = forward_runner
        if self.ranking_criterion in ['taylorfo'] or num_iterations > 1:
            assert self._trainer is not None
            assert self._optimizer is not None
        if self.ranking_criterion in ['l1_activation', 'l2_activation']:
            assert self._forward_runner is not None

        # Group generation: one group per attention layer, four weights per group
        self.masking_groups = []
        if self.attention_name_groups is not None:
            logger.info("Note: weights for the same attention layer are grouped using the given attention_name_groups.")
            self.group_weights_by_name()
        else:
            assert self.dummy_input is not None
            logger.info("Note: weights for the same attention layer are grouped using model graph.")
            self._unwrap_model()
            self.group_weight_names_by_graph()
            self._wrap_model()

        # Group sanity check
        self.validate_weight_groups()

        # Remove any mistakenly captured ungrouped modules
        self._unwrap_model()
        self.remove_ungrouped_modules()
        self._wrap_model()

        self.masker = MASKER_DICT[ranking_criterion](model, self, self.head_hidden_dim, **algo_kwargs)
        self.pruned_heads = {i: set() for i in range(len(self.masking_groups))}

    def group_weights_by_name(self):
        """
        Populate self.masking_groups using the groups specified by user in attention_name_groups.
        """
        assert len(self.masking_groups) == 0
        # build up masking groups
        name2group = {}
        for layer_idx, layer in enumerate(self.attention_name_groups):
            errmsg = 'Each name group must contain 4 weights, with the first three corresponding to Q_proj, K_proj, ' \
                     'V_proj (in any order) and the last one being output_proj.'
            assert len(layer) == 4, errmsg
            self.masking_groups.append([])
            for weight in layer:
                name2group[weight] = layer_idx

        # group wrappers
        for wrapper in self.get_modules_wrapper():
            if wrapper.name in name2group:
                wrapper.group_idx = name2group[wrapper.name]
                self.masking_groups[name2group[wrapper.name]].append(wrapper)

        logger.info('Grouping updated:')
        logger.info([[x.name for x in group] for group in self.masking_groups])

    def group_weight_names_by_graph(self):
        """
        Populate self.attention_name_groups by running inference on the module graph.
        Currently, the group inferred AttentionWeightDependency is limited to a set of four weights, with the first
        three corresponding to Q_proj, K_proj, V_proj (in any order) and the last one being output_proj.
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
            - head_hidden_dim must be a divisor of the output dimension of the projection weights (i.e., the resulting
              head number must be an integer)
        """
        errmsg = 'Attention weight group sanity check not passed'
        sparsity = None
        for group in self.masking_groups:
            # allow empty groups - may be caused by config list filtering
            if len(group) == 0:
                continue
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
            assert group[0].module.weight.size(0) % self.head_hidden_dim == 0, \
                errmsg + ': head_hidden_dim must be a divisor of the output dimension of the projection weights'

    def remove_ungrouped_modules(self):
        """
        Remove non-attention weights that might be mistakenly captured by a simplified config_list.
        Also update the corresponding list of layer information (self.modules_to_compress)
        """
        care_of_modules = set([x for layer in self.masking_groups for x in layer])

        modules_wrapper_new, modules_to_compress_new = [], []
        for wrapper, layer_info in zip(self.modules_wrapper, self.modules_to_compress):
            if wrapper in care_of_modules:
                modules_wrapper_new.append(wrapper)
                modules_to_compress_new.append(layer_info)

        self.modules_wrapper = modules_wrapper_new
        self.modules_to_compress = modules_to_compress_new

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
            if self.ranking_criterion in ['l1_activation', 'l2_activation']:
                training = self.bound_model.training
                self.bound_model.eval()
                self._forward_runner(self.bound_model)       # dry run, forward only
                self.update_mask()
                self.bound_model.train(training)
            elif self.ranking_criterion in ['taylorfo']:
                self._trainer(self.bound_model, optimizer=self._optimizer, criterion=self._criterion, epoch=None)
                self.update_mask()
            else:
                self.update_mask()

            # for iterative pruning, if not the last iteration, finetune before next iteration
            # Then, reset the maskers (may create additional hooks)
            if self.num_iterations > 1 and pruning_iter != self.num_iterations - 1:
                for e in range(self.epochs_per_iteration):
                    self._trainer(self.bound_model, optimizer=self._optimizer, criterion=self._criterion, epoch=e+1)
                self.masker.reset()

            logger.info('Pruned heads after iteration %i', pruning_iter)
            logger.info(self.pruned_heads)

    def update_mask(self):
        """
        Calculate and update masks for each masking group. If global_sort is set, the masks for all groups are
        calculated altogether, and then the groups are updated individually.
        """
        masks_for_all_groups = None
        if self.global_sort:
            masks_for_all_groups = self._calc_mask_global()
            assert len(masks_for_all_groups) == len(self.masking_groups)
        for group_idx, layer_weight_group in enumerate(self.masking_groups):
            if self.global_sort:
                masks = masks_for_all_groups[group_idx]
            else:
                masks = self._calc_mask(layer_weight_group)
            if masks is not None:
                for i, mask in enumerate(masks):
                    for mask_type in mask:
                        assert hasattr(layer_weight_group[i], mask_type), \
                            "there is no attribute '%s' in wrapper on %s" % (mask_type, layer_weight_group[i])
                        setattr(layer_weight_group[i], mask_type, mask[mask_type])
                        logger.debug(f'mask updated: {layer_weight_group[i].name} {mask_type}')

    def _calc_mask(self, weight_group):
        """
        Calculate mask for each group using only layer-local information.
        When global_sort is set for the pruner, _calc_mask_global should be called instead of this function.

        Parameters
        ----------
        weight_group : list
            A list of four wrappers generated by self.group_weights_by_name().

        Returns
        -------
        masks : list
            A four element list corresponding to the masks for each element in the four-element weight group.
            Each element in masks is a dict with keys "weight_mask" and "bias_mask" (optional).
            masks can be None if the underlying masker returns None. This means that the mask calculation fails.
            The calling function can try recalculate the mask at a later time. Note that the calling function might need
            to call masker.reset() before attempting to recalculate the mask.
        """
        iter_sparsity = weight_group[0].config['sparsity'] / self.num_iterations
        masks = self.masker.calc_mask(sparsity=iter_sparsity, weight_group=weight_group)

        return masks

    def _calc_mask_global(self):
        """
        Calculate mask for all groups using global information.

        Returns
        -------
        masks_list : list
            A list corresponding to the masks for each weight group in self.masking_groups. Each element in the
            returned mask_list is a four-element list corresponding to the masks for each element in a four-element
            weight group.
        """
        if len(self.get_modules_wrapper()) == 0:
            return []

        overall_sparsity = self.get_modules_wrapper()[0].config['sparsity'] / self.num_iterations
        n_heads_total = 0
        for group in self.masking_groups:
            if len(group) != 0:
                q_proj, _, _, _ = group
                n_heads_total += int(q_proj.module.weight.size()[0] / self.head_hidden_dim)
        n_heads_to_prune = int(n_heads_total * overall_sparsity)

        return self.masker.calc_mask_global(n_heads_to_prune)

    def calc_mask(self, wrapper, **kwargs):
        raise RuntimeError("Applications should directly call TransformerHeadPruner's update_mask() method.")
