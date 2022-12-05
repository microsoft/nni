# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# v2
from .attr import (
    get_nested_attr,
    set_nested_attr
)
from .config_validation import CompressorSchema
from .constructor_helper import (
    OptimizerConstructHelper,
    LRSchedulerConstructHelper
)
from .evaluator import (
    Evaluator,
    LightningEvaluator,
    TorchEvaluator,
    TransformersEvaluator,
    Hook,
    BackwardHook,
    ForwardHook,
    TensorHook
)
from .pruning import (
    config_list_canonical,
    unfold_config_list,
    dedupe_config_list,
    compute_sparsity_compact2origin,
    compute_sparsity_mask2compact,
    compute_sparsity,
    get_model_weights_numel,
    get_module_by_name,
    get_output_batch_dims
)
from .scaling import Scaling
from .check_ddp import (
    check_ddp_model,
    reset_ddp_model,
    all_reduce_on_multiple_gpus
)

# v1
from .counter import count_flops_params
from .mask_conflict import ChannelMaskConflict, GroupMaskConflict
from .utils import *
from .shape_dependency import *

def not_safe_to_prune(model, dummy_input):
    """
    Get the layers that are not safe to prune(may bring the shape conflict).
    For example, if the output tensor of a conv layer is directly followed by
    a shape-dependent function(such as reshape/view), then this conv layer
    may be not safe to be pruned. Pruning may change the output shape of
    this conv layer and result in shape problems. This function find all the
    layers that directly followed by the shape-dependent functions(view, reshape, etc).
    If you run the inference after the speedup and run into a shape related error,
    please exclude the layers returned by this function and try again.

    Parameters
    ----------
    model: torch.nn.Module
        The target model to prune.
    dummy_input: torch.Tensor/list of torch.Tensor/tuple of Tensor
    """
    reshape_dset = ReshapeDependency(model, dummy_input)
    return reshape_dset.dependency_sets
