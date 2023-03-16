# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict
import torch
import torch.nn as nn

def check_ddp_model(model: nn.Module):
    is_ddp_model = False
    ddp_params = {}
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        attr_dicts = model.__dict__

        ddp_params["device_ids"] = attr_dicts.get("device_ids", None)
        ddp_params["output_device"] = attr_dicts.get("output_device", None)
        ddp_params["dim"] = attr_dicts.get("dim", 0)
        ddp_params["broadcast_buffers"] = attr_dicts.get("broadcast_buffers", True)
        ddp_params["process_group"] = attr_dicts.get("process_group", None)
        ddp_params["bucket_cap_mb"] = attr_dicts.get("bucket_cap_mb", 25)
        ddp_params["find_unused_parameters"] = attr_dicts.get("find_unused_parameters", False)
        ddp_params["check_reduction"] = attr_dicts.get("check_reduction", False)

        # when torch version <= 1.6.0, there is no parameter "gradient_as_bucket_view"
        if "gradient_as_bucket_view" in attr_dicts:
            ddp_params["gradient_as_bucket_view"] = attr_dicts.get("gradient_as_bucket_view", False)
        # when torch version <= 1.10.0, there is no param "static_graph"
        if "static_graph" in attr_dicts:
            ddp_params["static_graph"] = attr_dicts.get("static_graph", False)

        is_ddp_model = True

    return is_ddp_model, ddp_params


def reset_ddp_model(model: torch.nn.parallel.DistributedDataParallel, ddp_params: Dict):
    assert isinstance(model, torch.nn.parallel.DistributedDataParallel)
    module = model.module
    return torch.nn.parallel.DistributedDataParallel(module=module, **ddp_params)
