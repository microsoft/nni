# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Entrypoint for trials.
"""

import nni
from .config.utils import parse_params
from .engine import CompressionEngine


if __name__ == '__main__':
    kwargs = nni.get_next_parameter()
    pruner_config, config_list, vessel, original_target, thetas = parse_params(kwargs)
    model, finetuner, evaluator, dummy_input, trainer, optimizer_helper, criterion, device = vessel.export()
    nni.report_final_result(0.0)
