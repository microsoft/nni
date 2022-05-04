# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Entrypoint for trials.
"""

import nni
from nni.algorithms.compression.v2.pytorch.pruning import PruningScheduler
from nni.algorithms.compression.v2.pytorch.pruning.tools import AGPTaskGenerator
from .config.utils import parse_params, parse_basic_pruner
from .engine import CompressionEngine


if __name__ == '__main__':
    kwargs = nni.get_next_parameter()
    pruner_config, config_list, vessel, original_target, thetas = parse_params(kwargs)
    basic_pruner, model, finetuner, evaluator, dummy_input, device = parse_basic_pruner(pruner_config, config_list, vessel)

    task_generator = AGPTaskGenerator(total_iteration=3, origin_model=model, origin_config_list=config_list)
    speedup = dummy_input is not None
    scheduler = PruningScheduler(pruner=basic_pruner, task_generator=task_generator, finetuner=finetuner, speedup=speedup,
                                 dummy_input=dummy_input, evaluator=evaluator)
    scheduler.compress()
    _, _, _, score, _ = scheduler.get_best_result()
    nni.report_final_result(score)
