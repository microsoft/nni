# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations
from typing import Any, Tuple, Dict, List, Type

from torch.nn import Module

from nni.compression.pytorch.utils import count_flops_params
from .compression import CompressionConfig
from .vessel import CompressionVessel


KEY_MODULE_NAME = 'module_name::'
KEY_PRUNERS = 'pruners'
KEY_VESSEL = '_vessel'
KEY_ORIGINAL_TARGET = '_original_target'
KEY_THETAS = '_thetas'


def _flops_theta_helper(target: int | float | str | None, origin: int) -> Tuple[float, float]:
    # hard code and magic number for flops/params reward function
    # the reward function is: sigmoid(flops_retained) = 1 / (1 + exp(-theta1 * (flops_retained + theta0)))
    # this helper function return a theta pair (theta0, theta1) for building a suitable (maybe) function.
    # the lower evaluating result (flops/params) compressed model has, the higher reward it gets.
    if not target or (isinstance(target, (int, float)) and target == 0):
        return (0., 0.)
    elif isinstance(target, float):
        assert 0. < target < 1.
        return (-0.1 - target, -50.)
    elif isinstance(target, int):
        assert 0 < target < origin
        return (-0.1 - target / origin, -50.)
    elif isinstance(target, str):
        raise NotImplementedError('Currently only supports setting the upper bound with int/float.')
    else:
        raise TypeError(f'Wrong target type: {type(target).__name__}, only support int/float/None.')


def _metric_theta_helper(target: float | None, origin: float) -> Tuple[float, float]:
    # hard code and magic number for metric reward function
    # only difference with `_flops_theta_helper` is the higher evaluating result (metric) compressed model has,
    # the higher reward it gets.
    if not target:
        return (-0.85, 50.)
    elif isinstance(target, float):
        assert 0. <= target <= 1.
        return (0.1 - target, 50.)
    else:
        raise TypeError(f'Wrong target type: {type(target).__name__}, only support float/None.')


def _summary_module_names(model: Module,
                          module_types: List[Type[Module] | str],
                          module_names: List[str],
                          exclude_module_names: List[str]) -> List[str]:
    # Return a list of module names that need to be compressed.
    # Include all names of modules that specified in `module_types` and `module_names` at first,
    # then remove the names specified in `exclude_module_names`.

    _module_types = set()
    _all_module_names = set()
    module_names_summary = set()
    if module_types:
        for module_type in module_types:
            if isinstance(module_type, Module):
                module_type = module_type.__name__
            assert isinstance(module_type, str)
            _module_types.add(module_type)

    # unfold module types as module names, add them to summary
    for module_name, module in model.named_modules():
        module_type = type(module).__name__
        if module_type in _module_types:
            module_names_summary.add(module_name)
        _all_module_names.add(module_name)

    # add module names to summary
    if module_names:
        for module_name in module_names:
            if module_name not in _all_module_names:
                # need warning, module_name not exist
                continue
            else:
                module_names_summary.add(module_name)

    # remove module names in exclude_module_names from module_names_summary
    if exclude_module_names:
        for module_name in exclude_module_names:
            if module_name not in _all_module_names:
                # need warning, module_name not exist
                continue
            if module_name in module_names_summary:
                module_names_summary.remove(module_name)

    return list(module_names_summary)


def generate_compression_search_space(config: CompressionConfig, vessel: CompressionVessel) -> Dict[str, Dict]:
    """
    Using config (constraints & priori) and vessel (model-related) to generate the hpo search space.
    """

    search_space = {}
    model, _, evaluator, dummy_input, _, _, _, _ = vessel.export()
    flops, params, results = count_flops_params(model, dummy_input, verbose=False, mode='full')
    metric = evaluator(model)

    module_names_summary = _summary_module_names(model, config.module_types, config.module_names, config.exclude_module_names)
    for module_name in module_names_summary:
        search_space['{}{}'.format(KEY_MODULE_NAME, module_name)] = {'_type': 'uniform', '_value': [0, 1]}

    assert not config.pruners or not config.quantizers

    # TODO: hard code for step 1, need refactor
    search_space[KEY_PRUNERS] = {'_type': 'choice', '_value': [pruner_config.json() for pruner_config in config.pruners]}

    original_target = {'flops': flops, 'params': params, 'metric': metric, 'results': results}

    # TODO: following fucntion need improvement
    flops_theta = _flops_theta_helper(config.flops, flops)
    params_theta = _flops_theta_helper(config.params, params)
    metric_theta = _metric_theta_helper(config.metric, metric)
    thetas = {'flops': flops_theta, 'params': params_theta, 'metric': metric_theta}

    search_space[KEY_VESSEL] = {'_type': 'choice', '_value': [vessel.json()]}
    search_space[KEY_ORIGINAL_TARGET] = {'_type': 'choice', '_value': [original_target]}
    search_space[KEY_THETAS] = {'_type': 'choice', '_value': [thetas]}
    return search_space


def parse_params(kwargs: Dict[str, Any]) -> Tuple[Dict[str, str], List[Dict[str, Any]], CompressionVessel, Dict[str, Any], Dict[str, Any]]:
    """
    Parse the parameters received by nni.get_next_parameter().

    Returns
    -------
    Dict[str, str], List[Dict[str, Any]], CompressionVessel, Dict[str, Any], Dict[str, Any]
        The compressor config, compressor config_list, model-related wrapper, evaluation value (flops, params, ...) for the original model,
        parameters of the hpo objective function.
    """
    compressor_config, vessel, original_target, thetas = None, None, None, None
    config_list = []

    for key, value in kwargs.items():
        if key.startswith(KEY_MODULE_NAME):
            config_list.append({'op_names': [key.split(KEY_MODULE_NAME)[1]], 'sparsity_per_layer': float(value)})
        elif key == KEY_PRUNERS:
            compressor_config = value
        elif key == KEY_VESSEL:
            vessel = CompressionVessel(**value)
        elif key == KEY_ORIGINAL_TARGET:
            original_target = value
        elif key == KEY_THETAS:
            thetas = value
        else:
            raise KeyError('Unrecognized key {}'.format(key))

    return compressor_config, config_list, vessel, original_target, thetas


def parse_basic_pruner(pruner_config: Dict[str, str], config_list: List[Dict[str, Any]], vessel: CompressionVessel):
    """
    Parse basic pruner and model-related objects used by pruning scheduler.
    """
    model, finetuner, evaluator, dummy_input, trainer, optimizer_helper, criterion, device = vessel.export()
    if pruner_config['pruner_type'] == 'L1NormPruner':
        from nni.compression.pytorch.pruning import L1NormPruner
        basic_pruner = L1NormPruner(model=model,
                                    config_list=config_list,
                                    mode=pruner_config['mode'],
                                    dummy_input=dummy_input)
    elif pruner_config['pruner_type'] == 'TaylorFOWeightPruner':
        from nni.compression.pytorch.pruning import TaylorFOWeightPruner
        basic_pruner = TaylorFOWeightPruner(model=model,
                                            config_list=config_list,
                                            trainer=trainer,
                                            traced_optimizer=optimizer_helper,
                                            criterion=criterion,
                                            training_batches=pruner_config['training_batches'],
                                            mode=pruner_config['mode'],
                                            dummy_input=dummy_input)
    else:
        raise ValueError('Unsupported basic pruner type {}'.format(pruner_config.pruner_type))
    return basic_pruner, model, finetuner, evaluator, dummy_input, device
