# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List
import uuid

import torch

from .shape_dependency import ChannelDependency, GroupDependency
from ..base.config import select_modules_by_config, trans_legacy_config_list


def auto_set_denpendency_group_ids(model: torch.nn.Module, config_list: List[Dict[str, Any]],
                                   dummy_input: torch.Tensor | List[torch.Tensor] | Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
    """
    Auto find the output dependency between all 'Conv2d', 'Linear', 'ConvTranspose2d', 'Embedding' modules,
    then set the ``dependency_group_id`` in config list.

    Note that a new dependency group id will be set as a shortcut in one config,
    it will replace the old configured one in that config.

    Parameters
    ----------
    model
        The origin model.
    config_list
        The compression config list.
    dummy_input
        The dummy input to the model forward function for tracing the model.
    """
    dependency = ChannelDependency(model, dummy_input)
    dependency.build_dependency()
    module2uid = {}
    for dependency_set in dependency.dependency_sets:
        uid = uuid.uuid4().hex
        module2uid.update({name: uid for name in dependency_set})

    group_dependency = GroupDependency(model, dummy_input)
    group_dependency.build_dependency()

    config_list = trans_legacy_config_list(config_list)
    new_config_list = []
    for config in config_list:
        modules, public_config, _ = select_modules_by_config(model, config)
        for name in modules.keys():
            sub_config = deepcopy(public_config)
            if name in module2uid:
                sub_config['dependency_group_id'] = module2uid[name]
            if name in group_dependency.dependency:
                sub_config['internal_metric_block'] = int(group_dependency.dependency[name])
            new_config_list.append({
                'op_names': [name],
                **sub_config
            })

    return new_config_list
