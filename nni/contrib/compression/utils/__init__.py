# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .check_ddp import check_ddp_model, reset_ddp_model
from .dependency import auto_set_denpendency_group_ids
from .evaluator import (Evaluator,
                        LightningEvaluator,
                        TorchEvaluator,
                        TransformersEvaluator,
                        DeepspeedTorchEvaluator)
from .scaling import Scaling
from .attr import (
    get_nested_attr,
    set_nested_attr,
    has_nested_attr
)
from .fused_utils import (
    validate_fused_modules_config,
    get_fused_module_list,
    update_config,
    check_bias
)

from .docstring import _EVALUATOR_DOCSTRING
