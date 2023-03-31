# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from nni.common.concrete_trace_utils import concrete_trace as nni_trace
from nni.compression.pytorch.speedup.v2 import ModelSpeedup
from nni.contrib.compression.pruning import L1NormPruner
from nni.contrib.compression.utils import auto_set_denpendency_group_ids


def speedup_pipeline(model):

    pruner = L1NormPruner(model, model.config_list)
    _, masks = pruner.compress()
    pruner.unwrap_model()

    if model.extra_info.need_auto_set_dependency:
        model.config_list = auto_set_denpendency_group_ids(model, model.config_list, tuple(model.dummy_inputs.values()))

    graph_module = trace_pipeline(model)

    ModelSpeedup(model, model.dummy_inputs, masks, graph_module=graph_module).speedup_model()

def trace_pipeline(model):
    if model.extra_info.need_run:
        for _ in range(2):
            getattr(model, model.extra_info.forward_function_name)(**model.dummy_inputs) # avoid partial init
    return nni_trace(model,
                     model.dummy_inputs,
                     leaf_module=model.extra_info.leaf_module,
                     fake_middle_class=model.extra_info.fake_middle_class,
                     forward_function_name=model.extra_info.forward_function_name)
