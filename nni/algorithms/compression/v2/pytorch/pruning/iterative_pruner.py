# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Dict, List, Callable, Optional

from torch import Tensor
from torch.nn import Module

from nni.algorithms.compression.v2.pytorch.utils import OptimizerConstructHelper

from .basic_pruner import (
    LevelPruner,
    L1NormPruner,
    L2NormPruner,
    FPGMPruner,
    SlimPruner,
    ActivationAPoZRankPruner,
    ActivationMeanRankPruner,
    TaylorFOWeightPruner,
    ADMMPruner
)
from .basic_scheduler import PruningScheduler
from .tools import (
    LinearTaskGenerator,
    AGPTaskGenerator,
    LotteryTicketTaskGenerator,
    SimulatedAnnealingTaskGenerator
)

_logger = logging.getLogger(__name__)

__all__ = ['LinearPruner', 'AGPPruner', 'LotteryTicketPruner', 'SimulatedAnnealingPruner']


PRUNER_DICT = {
    'level': LevelPruner,
    'l1': L1NormPruner,
    'l2': L2NormPruner,
    'fpgm': FPGMPruner,
    'slim': SlimPruner,
    'apoz': ActivationAPoZRankPruner,
    'mean_activation': ActivationMeanRankPruner,
    'taylorfo': TaylorFOWeightPruner,
    'admm': ADMMPruner
}


class IterativePruner(PruningScheduler):
    def _wrap_model(self):
        """
        Deprecated function.
        """
        _logger.warning('Nothing will happen when calling this function.\
            This pruner is an iterative pruner and does not directly wrap the model.')

    def _unwrap_model(self):
        """
        Deprecated function.
        """
        _logger.warning('Nothing will happen when calling this function.\
            This pruner is an iterative pruner and does not directly wrap the model.')

    def export_model(self, *args, **kwargs):
        """
        Deprecated function.
        """
        _logger.warning('Nothing will happen when calling this function.\
            The best result (and intermediate result if keeped) during iteration is under `log_dir` (default: \\.).')


class LinearPruner(IterativePruner):
    r"""
    Linear pruner is an iterative pruner, it will increase sparsity evenly from scratch during each iteration.

    For example, the final sparsity is set as 0.5, and the iteration number is 5, then the sparsity used in each iteration are ``[0, 0.1, 0.2, 0.3, 0.4, 0.5]``.

    Parameters
    ----------
    model : Module
        The origin unwrapped pytorch model to be pruned.
    config_list : List[Dict]
        The origin config list provided by the user.
    pruning_algorithm : str
        Supported pruning algorithm ['level', 'l1', 'l2', 'fpgm', 'slim', 'apoz', 'mean_activation', 'taylorfo', 'admm'].
        This iterative pruner will use the chosen corresponding pruner to prune the model in each iteration.
    total_iteration : int
        The total iteration number.
    log_dir : str
        The log directory use to saving the result, you can find the best result under this folder.
    keep_intermediate_result : bool
        If keeping the intermediate result, including intermediate model and masks during each iteration.
    finetuner : Optional[Callable[[Module], None]]
        The finetuner handled all finetune logic, use a pytorch module as input.
        It will be called at the end of each iteration, usually for neutralizing the accuracy loss brought by the pruning in this iteration.
    speedup : bool
        If set True, speedup the model at the end of each iteration to make the pruned model compact.
    dummy_input : Optional[torch.Tensor]
        If `speedup` is True, `dummy_input` is required for tracing the model in speedup.
    evaluator : Optional[Callable[[Module], float]]
        Evaluate the pruned model and give a score.
        If evaluator is None, the best result refers to the latest result.
    pruning_params : Dict
        If the chosen pruning_algorithm has extra parameters, put them as a dict to pass in.

    Examples
    --------
        >>> from nni.compression.pytorch.pruning import LinearPruner
        >>> config_list = [{'sparsity': 0.8, 'op_types': ['Conv2d']}]
        >>> finetuner = ...
        >>> pruner = LinearPruner(model, config_list, pruning_algorithm='l1', total_iteration=10, finetuner=finetuner)
        >>> pruner.compress()
        >>> _, model, masks, _, _ = pruner.get_best_result()

    For detailed example please refer to :githublink:`examples/model_compress/pruning/iterative_pruning_torch.py <examples/model_compress/pruning/iterative_pruning_torch.py>`
    """

    def __init__(self, model: Module, config_list: List[Dict], pruning_algorithm: str,
                 total_iteration: int, log_dir: str = '.', keep_intermediate_result: bool = False,
                 finetuner: Optional[Callable[[Module], None]] = None, speedup: bool = False, dummy_input: Optional[Tensor] = None,
                 evaluator: Optional[Callable[[Module], float]] = None, pruning_params: Dict = {}):
        task_generator = LinearTaskGenerator(total_iteration=total_iteration,
                                             origin_model=model,
                                             origin_config_list=config_list,
                                             log_dir=log_dir,
                                             keep_intermediate_result=keep_intermediate_result)
        if 'traced_optimizer' in pruning_params:
            pruning_params['traced_optimizer'] = OptimizerConstructHelper.from_trace(model, pruning_params['traced_optimizer'])
        pruner = PRUNER_DICT[pruning_algorithm](None, None, **pruning_params)
        super().__init__(pruner, task_generator, finetuner=finetuner, speedup=speedup, dummy_input=dummy_input,
                         evaluator=evaluator, reset_weight=False)


class AGPPruner(IterativePruner):
    r"""
    This is an iterative pruner, which the sparsity is increased from an initial sparsity value :math:`s_{i}` (usually 0) to a final sparsity value :math:`s_{f}` over a span of :math:`n` pruning iterations,
    starting at training step :math:`t_{0}` and with pruning frequency :math:`\Delta t`:

    :math:`s_{t}=s_{f}+\left(s_{i}-s_{f}\right)\left(1-\frac{t-t_{0}}{n \Delta t}\right)^{3} \text { for } t \in\left\{t_{0}, t_{0}+\Delta t, \ldots, t_{0} + n \Delta t\right\}`

    For more details please refer to `To prune, or not to prune: exploring the efficacy of pruning for model compression <https://arxiv.org/abs/1710.01878>`__\.

    Parameters
    ----------
    model : Module
        The origin unwrapped pytorch model to be pruned.
    config_list : List[Dict]
        The origin config list provided by the user.
    pruning_algorithm : str
        Supported pruning algorithm ['level', 'l1', 'l2', 'fpgm', 'slim', 'apoz', 'mean_activation', 'taylorfo', 'admm'].
        This iterative pruner will use the chosen corresponding pruner to prune the model in each iteration.
    total_iteration : int
        The total iteration number.
    log_dir : str
        The log directory use to saving the result, you can find the best result under this folder.
    keep_intermediate_result : bool
        If keeping the intermediate result, including intermediate model and masks during each iteration.
    finetuner : Optional[Callable[[Module], None]]
        The finetuner handled all finetune logic, use a pytorch module as input.
        It will be called at the end of each iteration, usually for neutralizing the accuracy loss brought by the pruning in this iteration.
    speedup : bool
        If set True, speedup the model at the end of each iteration to make the pruned model compact.
    dummy_input : Optional[torch.Tensor]
        If `speedup` is True, `dummy_input` is required for tracing the model in speedup.
    evaluator : Optional[Callable[[Module], float]]
        Evaluate the pruned model and give a score.
        If evaluator is None, the best result refers to the latest result.
    pruning_params : Dict
        If the chosen pruning_algorithm has extra parameters, put them as a dict to pass in.

    Examples
    --------
        >>> from nni.compression.pytorch.pruning import AGPPruner
        >>> config_list = [{'sparsity': 0.8, 'op_types': ['Conv2d']}]
        >>> finetuner = ...
        >>> pruner = AGPPruner(model, config_list, pruning_algorithm='l1', total_iteration=10, finetuner=finetuner)
        >>> pruner.compress()
        >>> _, model, masks, _, _ = pruner.get_best_result()

    For detailed example please refer to :githublink:`examples/model_compress/pruning/iterative_pruning_torch.py <examples/model_compress/pruning/iterative_pruning_torch.py>`
    """

    def __init__(self, model: Module, config_list: List[Dict], pruning_algorithm: str,
                 total_iteration: int, log_dir: str = '.', keep_intermediate_result: bool = False,
                 finetuner: Optional[Callable[[Module], None]] = None, speedup: bool = False, dummy_input: Optional[Tensor] = None,
                 evaluator: Optional[Callable[[Module], float]] = None, pruning_params: Dict = {}):
        task_generator = AGPTaskGenerator(total_iteration=total_iteration,
                                          origin_model=model,
                                          origin_config_list=config_list,
                                          log_dir=log_dir,
                                          keep_intermediate_result=keep_intermediate_result)
        if 'traced_optimizer' in pruning_params:
            pruning_params['traced_optimizer'] = OptimizerConstructHelper.from_trace(model, pruning_params['traced_optimizer'])
        pruner = PRUNER_DICT[pruning_algorithm](None, None, **pruning_params)
        super().__init__(pruner, task_generator, finetuner=finetuner, speedup=speedup, dummy_input=dummy_input,
                         evaluator=evaluator, reset_weight=False)


class LotteryTicketPruner(IterativePruner):
    r"""
    `The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks <https://arxiv.org/abs/1803.03635>`__\ ,
    authors Jonathan Frankle and Michael Carbin,provides comprehensive measurement and analysis,
    and articulate the *lottery ticket hypothesis*\ : dense, randomly-initialized, feed-forward networks contain subnetworks (*winning tickets*\ ) that
    -- when trained in isolation -- reach test accuracy comparable to the original network in a similar number of iterations.

    In this paper, the authors use the following process to prune a model, called *iterative prunning*\ :

    ..

        #. Randomly initialize a neural network f(x;theta_0) (where theta\ *0 follows D*\ {theta}).
        #. Train the network for j iterations, arriving at parameters theta_j.
        #. Prune p% of the parameters in theta_j, creating a mask m.
        #. Reset the remaining parameters to their values in theta_0, creating the winning ticket f(x;m*theta_0).
        #. Repeat step 2, 3, and 4.

    If the configured final sparsity is P (e.g., 0.8) and there are n times iterative pruning,
    each iterative pruning prunes 1-(1-P)^(1/n) of the weights that survive the previous round.

    Parameters
    ----------
    model : Module
        The origin unwrapped pytorch model to be pruned.
    config_list : List[Dict]
        The origin config list provided by the user.
    pruning_algorithm : str
        Supported pruning algorithm ['level', 'l1', 'l2', 'fpgm', 'slim', 'apoz', 'mean_activation', 'taylorfo', 'admm'].
        This iterative pruner will use the chosen corresponding pruner to prune the model in each iteration.
    total_iteration : int
        The total iteration number.
    log_dir : str
        The log directory use to saving the result, you can find the best result under this folder.
    keep_intermediate_result : bool
        If keeping the intermediate result, including intermediate model and masks during each iteration.
    finetuner : Optional[Callable[[Module], None]]
        The finetuner handled all finetune logic, use a pytorch module as input.
        It will be called at the end of each iteration if reset_weight is False, will be called at the beginning of each iteration otherwise.
    speedup : bool
        If set True, speedup the model at the end of each iteration to make the pruned model compact.
    dummy_input : Optional[torch.Tensor]
        If `speedup` is True, `dummy_input` is required for tracing the model in speedup.
    evaluator : Optional[Callable[[Module], float]]
        Evaluate the pruned model and give a score.
        If evaluator is None, the best result refers to the latest result.
    reset_weight : bool
        If set True, the model weight will reset to the original model weight at the end of each iteration step.
    pruning_params : Dict
        If the chosen pruning_algorithm has extra parameters, put them as a dict to pass in.

    Examples
    --------
        >>> from nni.compression.pytorch.pruning import LotteryTicketPruner
        >>> config_list = [{'sparsity': 0.8, 'op_types': ['Conv2d']}]
        >>> finetuner = ...
        >>> pruner = LotteryTicketPruner(model, config_list, pruning_algorithm='l1', total_iteration=10, finetuner=finetuner, reset_weight=True)
        >>> pruner.compress()
        >>> _, model, masks, _, _ = pruner.get_best_result()

    For detailed example please refer to :githublink:`examples/model_compress/pruning/iterative_pruning_torch.py <examples/model_compress/pruning/iterative_pruning_torch.py>`

    """

    def __init__(self, model: Module, config_list: List[Dict], pruning_algorithm: str,
                 total_iteration: int, log_dir: str = '.', keep_intermediate_result: bool = False,
                 finetuner: Optional[Callable[[Module], None]] = None, speedup: bool = False, dummy_input: Optional[Tensor] = None,
                 evaluator: Optional[Callable[[Module], float]] = None, reset_weight: bool = True,
                 pruning_params: Dict = {}):
        task_generator = LotteryTicketTaskGenerator(total_iteration=total_iteration,
                                                    origin_model=model,
                                                    origin_config_list=config_list,
                                                    log_dir=log_dir,
                                                    keep_intermediate_result=keep_intermediate_result)
        if 'traced_optimizer' in pruning_params:
            pruning_params['traced_optimizer'] = OptimizerConstructHelper.from_trace(model, pruning_params['traced_optimizer'])
        pruner = PRUNER_DICT[pruning_algorithm](None, None, **pruning_params)
        super().__init__(pruner, task_generator, finetuner=finetuner, speedup=speedup, dummy_input=dummy_input,
                         evaluator=evaluator, reset_weight=reset_weight)


class SimulatedAnnealingPruner(IterativePruner):
    """
    We implement a guided heuristic search method, Simulated Annealing (SA) algorithm. As mentioned in the paper, this method is enhanced on guided search based on prior experience.
    The enhanced SA technique is based on the observation that a DNN layer with more number of weights often has a higher degree of model compression with less impact on overall accuracy.

    * Randomly initialize a pruning rate distribution (sparsities).
    * While current_temperature < stop_temperature:

        #. generate a perturbation to current distribution
        #. Perform fast evaluation on the perturbated distribution
        #. accept the perturbation according to the performance and probability, if not accepted, return to step 1
        #. cool down, current_temperature <- current_temperature * cool_down_rate

    For more details, please refer to `AutoCompress: An Automatic DNN Structured Pruning Framework for Ultra-High Compression Rates <https://arxiv.org/abs/1907.03141>`__.

    Parameters
    ----------
    model : Module
        The origin unwrapped pytorch model to be pruned.
    config_list : List[Dict]
        The origin config list provided by the user.
    evaluator : Callable[[Module], float]
        Evaluate the pruned model and give a score.
    start_temperature : float
        Start temperature of the simulated annealing process.
    stop_temperature : float
        Stop temperature of the simulated annealing process.
    cool_down_rate : float
        Cool down rate of the temperature.
    perturbation_magnitude : float
        Initial perturbation magnitude to the sparsities. The magnitude decreases with current temperature.
    pruning_algorithm : str
        Supported pruning algorithm ['level', 'l1', 'l2', 'fpgm', 'slim', 'apoz', 'mean_activation', 'taylorfo', 'admm'].
        This iterative pruner will use the chosen corresponding pruner to prune the model in each iteration.
    pruning_params : Dict
        If the chosen pruning_algorithm has extra parameters, put them as a dict to pass in.
    log_dir : str
        The log directory use to saving the result, you can find the best result under this folder.
    keep_intermediate_result : bool
        If keeping the intermediate result, including intermediate model and masks during each iteration.
    finetuner : Optional[Callable[[Module], None]]
        The finetuner handled all finetune logic, use a pytorch module as input, will be called in each iteration.
    speedup : bool
        If set True, speedup the model at the end of each iteration to make the pruned model compact.
    dummy_input : Optional[torch.Tensor]
        If `speedup` is True, `dummy_input` is required for tracing the model in speedup.

    Examples
    --------
        >>> from nni.compression.pytorch.pruning import SimulatedAnnealingPruner
        >>> model = ...
        >>> config_list = [{'sparsity': 0.8, 'op_types': ['Conv2d']}]
        >>> evaluator = ...
        >>> finetuner = ...
        >>> pruner = SimulatedAnnealingPruner(model, config_list, pruning_algorithm='l1', evaluator=evaluator, cool_down_rate=0.9, finetuner=finetuner)
        >>> pruner.compress()
        >>> _, model, masks, _, _ = pruner.get_best_result()

    For detailed example please refer to :githublink:`examples/model_compress/pruning/simulated_anealing_pruning_torch.py <examples/model_compress/pruning/simulated_anealing_pruning_torch.py>`
    """

    def __init__(self, model: Module, config_list: List[Dict], evaluator: Callable[[Module], float], start_temperature: float = 100,
                 stop_temperature: float = 20, cool_down_rate: float = 0.9, perturbation_magnitude: float = 0.35,
                 pruning_algorithm: str = 'level', pruning_params: Dict = {}, log_dir: str = '.', keep_intermediate_result: bool = False,
                 finetuner: Optional[Callable[[Module], None]] = None, speedup: bool = False, dummy_input: Optional[Tensor] = None):
        task_generator = SimulatedAnnealingTaskGenerator(origin_model=model,
                                                         origin_config_list=config_list,
                                                         start_temperature=start_temperature,
                                                         stop_temperature=stop_temperature,
                                                         cool_down_rate=cool_down_rate,
                                                         perturbation_magnitude=perturbation_magnitude,
                                                         log_dir=log_dir,
                                                         keep_intermediate_result=keep_intermediate_result)
        if 'traced_optimizer' in pruning_params:
            pruning_params['traced_optimizer'] = OptimizerConstructHelper.from_trace(model, pruning_params['traced_optimizer'])
        pruner = PRUNER_DICT[pruning_algorithm](None, None, **pruning_params)
        super().__init__(pruner, task_generator, finetuner=finetuner, speedup=speedup, dummy_input=dummy_input,
                         evaluator=evaluator, reset_weight=False)
