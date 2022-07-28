# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Callable, Optional, overload

from torch import Tensor
from torch.nn import Module

from .basic_pruner import ADMMPruner
from .iterative_pruner import IterativePruner, SimulatedAnnealingPruner
from .tools import LotteryTicketTaskGenerator
from ..utils import LightningEvaluator, TorchEvaluator, OptimizerConstructHelper
from ..utils.docstring import _EVALUATOR_DOCSTRING

_logger = logging.getLogger(__name__)


class AutoCompressTaskGenerator(LotteryTicketTaskGenerator):
    def __init__(self, total_iteration: int, origin_model: Module, origin_config_list: List[Dict],
                 origin_masks: Dict[str, Dict[str, Tensor]] = {}, sa_params: Dict = {}, log_dir: str = '.',
                 keep_intermediate_result: bool = False):
        self._sa_params = sa_params
        super().__init__(total_iteration=total_iteration,
                         origin_model=origin_model,
                         origin_config_list=origin_config_list,
                         origin_masks=origin_masks,
                         log_dir=log_dir,
                         keep_intermediate_result=keep_intermediate_result)

    def reset(self, model: Module, config_list: List[Dict] = [], masks: Dict[str, Dict[str, Tensor]] = {}):
        # TODO: replace with validation here
        for config in config_list:
            if 'sparsity' in config or 'sparsity_per_layer' in config:
                warn_msg = 'Only `total_sparsity` can be differentially allocated sparse ratio to each layer, ' + \
                           '`sparsity` or `sparsity_per_layer` will allocate fixed sparse ratio to layers. ' + \
                           'Make sure you know what this will lead to, otherwise please use `total_sparsity`.'
                _logger.warning(warn_msg)
        return super().reset(model, config_list, masks)

    def _iterative_pruner_reset(self, model: Module, config_list: List[Dict] = [], masks: Dict[str, Dict[str, Tensor]] = {}):
        if not hasattr(self, 'iterative_pruner'):
            self.iterative_pruner = SimulatedAnnealingPruner(model=model,
                                                             config_list=config_list,
                                                             log_dir=Path(self._log_dir_root, 'SA'),
                                                             **self._sa_params)
        else:
            self.iterative_pruner.reset(model, config_list=config_list, masks=masks)

    def allocate_sparsity(self, new_config_list: List[Dict], model: Module, masks: Dict[str, Dict[str, Tensor]]):
        self._iterative_pruner_reset(model, new_config_list, masks)
        self.iterative_pruner.compress()
        best_result = self.iterative_pruner.get_best_result()
        assert best_result is not None, 'Best result does not exist, iterative pruner may not start pruning.'
        _, _, _, _, config_list = best_result
        return config_list


class AutoCompressPruner(IterativePruner):
    __doc__ = r"""
    For total iteration number :math:`N`, AutoCompressPruner prune the model that survive the previous iteration for a fixed sparsity ratio (e.g., :math:`1-{(1-0.8)}^{(1/N)}`) to achieve the overall sparsity (e.g., :math:`0.8`):
    """ + r"""

    .. code-block:: bash

        1. Generate sparsities distribution using SimulatedAnnealingPruner
        2. Perform ADMM-based pruning to generate pruning result for the next iteration.

    For more details, please refer to `AutoCompress: An Automatic DNN Structured Pruning Framework for Ultra-High Compression Rates <https://arxiv.org/abs/1907.03141>`__.

    Parameters
    ----------
    model
        The origin unwrapped pytorch model to be pruned.
    config_list
        The origin config list provided by the user.
    total_iteration
        The total iteration number.
    admm_params
        The parameters passed to the ADMMPruner.

        - evaluator : LightningEvaluator or TorchEvaluator.
            The same with the evaluator of AutoCompressPruner input parameter.
        - iterations : int.
            The total iteration number in admm pruning algorithm.
        - training_epochs : int.
            The epoch number for training model in each iteration.

    sa_params
        The parameters passed to the SimulatedAnnealingPruner.

        - evaluator : LightningEvaluator or TorchEvaluator.
            The same with the evaluator of AutoCompressPruner input parameter.
        - start_temperature : float. Default: `100`.
            Start temperature of the simulated annealing process.
        - stop_temperature : float. Default: `20`.
            Stop temperature of the simulated annealing process.
        - cool_down_rate : float. Default: `0.9`.
            Cooldown rate of the temperature.
        - perturbation_magnitude : float. Default: `0.35`.
            Initial perturbation magnitude to the sparsities. The magnitude decreases with current temperature.
        - pruning_algorithm : str. Default: `'level'`.
            Supported pruning algorithm ['level', 'l1', 'l2', 'fpgm', 'slim', 'apoz', 'mean_activation', 'taylorfo', 'admm'].
        - pruning_params : Dict. Default: dict().
            If the chosen pruning_algorithm has extra parameters, put them as a dict to pass in.

    log_dir
        The log directory used to save the result, you can find the best result under this folder.
    keep_intermediate_result
        If keeping the intermediate result, including intermediate model and masks during each iteration.
    evaluator
        ``evaluator`` is used to replace the previous ``finetuner``, ``dummy_input`` and old ``evaluator`` API.
        {evaluator_docstring}
        The old API (``finetuner``, ``dummy_input`` and old ``evaluator``) is still supported and will be deprecated in v3.0.
        If you want to consult the old API, please refer to `v2.8 pruner API <https://nni.readthedocs.io/en/v2.8/reference/compression/pruner.html>`__.
    speedup
        If set True, speedup the model at the end of each iteration to make the pruned model compact.

    Notes
    -----
    The full script can be found :githublink:`here <examples/model_compress/pruning/auto_compress_pruner.py>`.
    """.format(evaluator_docstring=_EVALUATOR_DOCSTRING)

    @overload
    def __init__(self, model: Module, config_list: List[Dict], total_iteration: int, admm_params: Dict,
                 sa_params: Dict, log_dir: str = '.', keep_intermediate_result: bool = False,
                 evaluator: LightningEvaluator | TorchEvaluator | None = None, speedup: bool = False):
        ...

    @overload
    def __init__(self, model: Module, config_list: List[Dict], total_iteration: int, admm_params: Dict,
                 sa_params: Dict, log_dir: str = '.', keep_intermediate_result: bool = False,
                 finetuner: Optional[Callable[[Module], None]] = None, speedup: bool = False,
                 dummy_input: Optional[Tensor] = None, evaluator: Optional[Callable[[Module], float]] = None):
        ...

    def __init__(self, model: Module, config_list: List[Dict], total_iteration: int, admm_params: Dict,
                 sa_params: Dict, log_dir: str = '.', keep_intermediate_result: bool = False,
                 *args, **kwargs):
        new_api = ['evaluator', 'speedup']
        new_init_kwargs = {'evaluator': None, 'speedup': False}
        old_api = ['finetuner', 'speedup', 'dummy_input', 'evaluator']
        old_init_kwargs = {'finetuner': None, 'evaluator': None, 'dummy_input': None, 'speedup': False}
        init_kwargs = self._init_evaluator(model, new_api, new_init_kwargs, old_api, old_init_kwargs, args, kwargs)

        speedup = init_kwargs['speedup']

        task_generator = AutoCompressTaskGenerator(total_iteration=total_iteration,
                                                   origin_model=model,
                                                   origin_config_list=config_list,
                                                   sa_params=sa_params,
                                                   log_dir=log_dir,
                                                   keep_intermediate_result=keep_intermediate_result)
        if 'traced_optimizer' in admm_params:
            admm_params['traced_optimizer'] = OptimizerConstructHelper.from_trace(model, admm_params['traced_optimizer'])
        # granularity in ADMM stage will align with SA stage, if 'granularity' is not specify
        if 'granularity' not in admm_params:
            # only if level pruning and fine-grained admm pruning used in SA, fine-grained admm pruning will used in auto-compress
            if 'pruning_algorithm' in sa_params:
                sa_algo = sa_params['pruning_algorithm']
                sa_algo_params = sa_params.get('pruning_params')
                if sa_algo in ['level']:
                    admm_params['granularity'] = 'fine-grained'
                elif sa_algo in ['admm'] and (sa_algo_params is not None) and not (sa_algo_params.get('granularity') == 'coarse-grained'):
                    admm_params['granularity'] = 'fine-grained'
                else:
                    admm_params['granularity'] = 'coarse-grained'
            else:
                admm_params['granularity'] = 'fine-grained'

        pruner = ADMMPruner(None, None, **admm_params)  # type: ignore

        if self.using_evaluator:
            super().__init__(pruner, task_generator, evaluator=self.evaluator, speedup=speedup, reset_weight=False)
        else:
            super().__init__(pruner, task_generator, finetuner=self.finetuner, speedup=speedup, dummy_input=self.dummy_input,
                             evaluator=self._evaluator, reset_weight=False)  # type: ignore
