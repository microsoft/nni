# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from pathlib import Path
from typing import Dict, List, Callable, Optional

from torch import Tensor
from torch.nn import Module

from nni.algorithms.compression.v2.pytorch.utils import OptimizerConstructHelper

from .basic_pruner import ADMMPruner
from .iterative_pruner import IterativePruner, SimulatedAnnealingPruner
from .tools import LotteryTicketTaskGenerator

_logger = logging.getLogger(__name__)


class AutoCompressTaskGenerator(LotteryTicketTaskGenerator):
    def __init__(self, total_iteration: int, origin_model: Module, origin_config_list: List[Dict],
                 origin_masks: Dict[str, Dict[str, Tensor]] = {}, sa_params: Dict = {}, log_dir: str = '.',
                 keep_intermediate_result: bool = False):
        self.iterative_pruner = SimulatedAnnealingPruner(model=None,
                                                         config_list=None,
                                                         log_dir=Path(log_dir, 'SA'),
                                                         **sa_params)
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
                _logger.warning('Only `total_sparsity` can be differentially allocated sparse ratio to each layer, `sparsity` or `sparsity_per_layer` will allocate fixed sparse ratio to layers. Make sure you know what this will lead to, otherwise please use `total_sparsity`.')
        return super().reset(model, config_list, masks)

    def _iterative_pruner_reset(self, model: Module, config_list: List[Dict] = [], masks: Dict[str, Dict[str, Tensor]] = {}):
        self.iterative_pruner.task_generator._log_dir = Path(self._log_dir_root, 'SA')
        self.iterative_pruner.reset(model, config_list=config_list, masks=masks)

    def allocate_sparsity(self, new_config_list: List[Dict], model: Module, masks: Dict[str, Dict[str, Tensor]]):
        self._iterative_pruner_reset(model, new_config_list, masks)
        self.iterative_pruner.compress()
        _, _, _, _, config_list = self.iterative_pruner.get_best_result()
        return config_list


class AutoCompressPruner(IterativePruner):
    r"""
    For total iteration number :math:`N`, AutoCompressPruner prune the model that survive the previous iteration for a fixed sparsity ratio (e.g., :math:`1-{(1-0.8)}^{(1/N)}`) to achieve the overall sparsity (e.g., :math:`0.8`):

    .. code-block:: bash

        1. Generate sparsities distribution using SimulatedAnnealingPruner
        2. Perform ADMM-based pruning to generate pruning result for the next iteration.

    For more details, please refer to `AutoCompress: An Automatic DNN Structured Pruning Framework for Ultra-High Compression Rates <https://arxiv.org/abs/1907.03141>`__.

    Parameters
    ----------
    model : Module
        The origin unwrapped pytorch model to be pruned.
    config_list : List[Dict]
        The origin config list provided by the user.
    total_iteration : int
        The total iteration number.
    evaluator : Callable[[Module], float]
        Evaluate the pruned model and give a score.
    admm_params : Dict
        The parameters passed to the ADMMPruner.

        - trainer : Callable[[Module, Optimizer, Callable].
            A callable function used to train model or just inference. Take model, optimizer, criterion as input.
            The model will be trained or inferenced `training_epochs` epochs.
        - traced_optimizer : nni.common.serializer.Traceable(torch.optim.Optimizer)
            The traced optimizer instance which the optimizer class is wrapped by nni.trace.
            E.g. ``traced_optimizer = nni.trace(torch.nn.Adam)(model.parameters())``.
        - criterion : Callable[[Tensor, Tensor], Tensor].
            The criterion function used in trainer. Take model output and target value as input, and return the loss.
        - iterations : int.
            The total iteration number in admm pruning algorithm.
        - training_epochs : int.
            The epoch number for training model in each iteration.

    sa_params : Dict
        The parameters passed to the SimulatedAnnealingPruner.

        - evaluator : Callable[[Module], float]. Required.
            Evaluate the pruned model and give a score.
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
        - pruning_params : Dict. Default: `{}`.
            If the chosen pruning_algorithm has extra parameters, put them as a dict to pass in.

    log_dir : str
        The log directory used to save the result, you can find the best result under this folder.
    keep_intermediate_result : bool
        If keeping the intermediate result, including intermediate model and masks during each iteration.
    finetuner : Optional[Callable[[Module], None]]
        The finetuner handles all finetune logic, takes a pytorch module as input.
        It will be called at the end of each iteration, usually for neutralizing the accuracy loss brought by the pruning in this iteration.
    speedup : bool
        If set True, speedup the model at the end of each iteration to make the pruned model compact.
    dummy_input : Optional[torch.Tensor]
        If `speedup` is True, `dummy_input` is required for tracing the model in speedup.

    Examples
    --------
        >>> import nni
        >>> from nni.compression.pytorch.pruning import AutoCompressPruner
        >>> model = ...
        >>> config_list = [{ 'sparsity': 0.8, 'op_types': ['Conv2d'] }]
        >>> # make sure you have used nni.trace to wrap the optimizer class before initialize
        >>> traced_optimizer = nni.trace(torch.optim.Adam)(model.parameters())
        >>> trainer = ...
        >>> criterion = ...
        >>> evaluator = ...
        >>> finetuner = ...
        >>> admm_params = {
        >>>     'trainer': trainer,
        >>>     'traced_optimizer': traced_optimizer,
        >>>     'criterion': criterion,
        >>>     'iterations': 10,
        >>>     'training_epochs': 1
        >>> }
        >>> sa_params = {
        >>>     'evaluator': evaluator
        >>> }
        >>> pruner = AutoCompressPruner(model, config_list, 10, admm_params, sa_params, finetuner=finetuner)
        >>> pruner.compress()
        >>> _, model, masks, _, _ = pruner.get_best_result()

    The full script can be found :githublink:`here <examples/model_compress/pruning/auto_compress_pruner.py>`.
    """

    def __init__(self, model: Module, config_list: List[Dict], total_iteration: int, admm_params: Dict,
                 sa_params: Dict, log_dir: str = '.', keep_intermediate_result: bool = False,
                 finetuner: Optional[Callable[[Module], None]] = None, speedup: bool = False,
                 dummy_input: Optional[Tensor] = None, evaluator: Callable[[Module], float] = None):
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

        pruner = ADMMPruner(None, None, **admm_params)
        super().__init__(pruner, task_generator, finetuner=finetuner, speedup=speedup, dummy_input=dummy_input,
                         evaluator=evaluator, reset_weight=False)
