# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Entrypoint for trials.
TODO: split this file to several modules
"""

import math
import os
from pathlib import Path

import nni
from nni.algorithms.compression.v2.pytorch.pruning import PruningScheduler
from nni.algorithms.compression.v2.pytorch.pruning.tools import AGPTaskGenerator
from nni.compression.pytorch.utils import count_flops_params
from .config.utils import parse_params, parse_basic_pruner

# TODO: move this function to evaluate module
def sigmoid(x: float, theta0: float = -0.5, theta1: float = 10) -> float:
    return 1 / (1 + math.exp(-theta1 * (x + theta0)))

if __name__ == '__main__':
    kwargs = nni.get_next_parameter()
    pruner_config, config_list, vessel, original_target, thetas = parse_params(kwargs)
    basic_pruner, model, finetuner, evaluator, dummy_input, device = parse_basic_pruner(pruner_config, config_list, vessel)

    # TODO: move following logic to excution engine
    log_dir = Path(os.environ['NNI_OUTPUT_DIR']) if 'NNI_OUTPUT_DIR' in os.environ else Path('nni_outputs', 'log')
    task_generator = AGPTaskGenerator(total_iteration=3, origin_model=model, origin_config_list=config_list,
                                      skip_first_iteration=True, log_dir=log_dir)
    speedup = dummy_input is not None
    scheduler = PruningScheduler(pruner=basic_pruner, task_generator=task_generator, finetuner=finetuner, speedup=speedup,
                                 dummy_input=dummy_input, evaluator=None)
    scheduler.compress()
    _, model, _, _, _ = scheduler.get_best_result()
    metric = evaluator(model)
    flops, params, _ = count_flops_params(model, dummy_input, verbose=False, mode='full')

    # TODO: more efficient way to calculate or combine these scores
    flops_score = sigmoid(flops / original_target['flops'], *thetas['flops'])
    params_score = sigmoid(params / original_target['params'], *thetas['params'])
    metric_score = sigmoid(metric / original_target['metric'], *thetas['metric'])
    final_result = flops_score + params_score + metric_score

    nni.report_final_result({'default': final_result, 'flops': flops, 'params': params, 'metric': metric})
