# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import pytest

from nni.compression.pytorch.pruning import (
    LinearPruner,
    AGPPruner,
    LotteryTicketPruner,
    SimulatedAnnealingPruner,
    AutoCompressPruner,
    AMCPruner
)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

from assets.common import create_model, log_dir, validate_masks, validate_dependency_aware
from assets.device import device
from assets.simple_mnist import (
    create_lighting_evaluator,
    create_pytorch_evaluator,
    training_model,
    finetuning_model,
    evaluating_model
)


@pytest.mark.parametrize('model_type', ['lightning', 'pytorch'])
@pytest.mark.parametrize('using_evaluator', [True, False])
def test_linear_pruner(model_type: str, using_evaluator: bool):
    model, config_list, dummy_input = create_model(model_type)

    if using_evaluator:
        evaluator = create_lighting_evaluator() if model_type == 'lightning' else create_pytorch_evaluator(model)
        pruner = LinearPruner(model=model, config_list=config_list, pruning_algorithm='l1', total_iteration=2,
                              log_dir=log_dir, keep_intermediate_result=False, evaluator=evaluator, speedup=True,
                              pruning_params={'mode': 'dependency_aware', 'dummy_input': dummy_input})
    else:
        model.to(device)
        dummy_input = dummy_input.to(device)
        pruner = LinearPruner(model=model, config_list=config_list, pruning_algorithm='l1', total_iteration=2, log_dir=log_dir,
                              keep_intermediate_result=False, finetuner=finetuning_model, speedup=True, dummy_input=dummy_input,
                              evaluator=None, pruning_params={'mode': 'dependency_aware', 'dummy_input': dummy_input})

    pruner.compress()
    best_task_id, best_model, best_masks, best_score, best_config_list = pruner.get_best_result()
    model(dummy_input)
    pruner._unwrap_model()
    validate_masks(best_masks, best_model, config_list)
