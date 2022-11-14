# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, overload

from torch import Tensor
from torch.nn import Module

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
from .basic_scheduler import PruningScheduler, _LEGACY_FINETUNER, _LEGACY_EVALUATOR
from .tools import (
    LinearTaskGenerator,
    AGPTaskGenerator,
    LotteryTicketTaskGenerator,
    SimulatedAnnealingTaskGenerator
)
from ..utils import (
    OptimizerConstructHelper,
    Evaluator
)
from ..utils.docstring import _EVALUATOR_DOCSTRING

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
    __doc__ = r"""
    Linear pruner is an iterative pruner, it will increase sparsity evenly from scratch during each iteration.

    For example, the final sparsity is set as 0.5, and the iteration number is 5, then the sparsity used in each iteration are ``[0, 0.1, 0.2, 0.3, 0.4, 0.5]``.

    Parameters
    ----------
    model
        The origin unwrapped pytorch model to be pruned.
    config_list
        The origin config list provided by the user.
    pruning_algorithm
        Supported pruning algorithm ['level', 'l1', 'l2', 'fpgm', 'slim', 'apoz', 'mean_activation', 'taylorfo', 'admm'].
        This iterative pruner will use the chosen corresponding pruner to prune the model in each iteration.
    total_iteration
        The total iteration number.
    log_dir
        The log directory use to saving the result, you can find the best result under this folder.
    keep_intermediate_result
        If keeping the intermediate result, including intermediate model and masks during each iteration.
    evaluator
        ``evaluator`` is used to replace the previous ``finetuner``, ``dummy_input`` and old ``evaluator`` API.
        {evaluator_docstring}
        The old API (``finetuner``, ``dummy_input`` and old ``evaluator``) is still supported and will be deprecated in v3.0.
        If you want to consult the old API, please refer to `v2.8 pruner API <https://nni.readthedocs.io/en/v2.8/reference/compression/pruner.html>`__.
    speedup
        If set True, speedup the model at the end of each iteration to make the pruned model compact.
    pruning_params
        If the chosen pruning_algorithm has extra parameters, put them as a dict to pass in.

    Notes
    -----
    For detailed example please refer to :githublink:`examples/model_compress/pruning/iterative_pruning_torch.py <examples/model_compress/pruning/iterative_pruning_torch.py>`
    """.format(evaluator_docstring=_EVALUATOR_DOCSTRING)

    @overload
    def __init__(self, model: Module, config_list: List[Dict], pruning_algorithm: str,
                 total_iteration: int, log_dir: str = '.', keep_intermediate_result: bool = False,
                 evaluator: Evaluator | None = None, speedup: bool = False,
                 pruning_params: Dict = {}):
        ...

    @overload
    def __init__(self, model: Module, config_list: List[Dict], pruning_algorithm: str,
                 total_iteration: int, log_dir: str = '.', keep_intermediate_result: bool = False,
                 finetuner: _LEGACY_FINETUNER | None = None, speedup: bool = False, dummy_input: Any | None = None,
                 evaluator: _LEGACY_EVALUATOR | None = None, pruning_params: Dict = {}):
        ...

    def __init__(self, model: Module, config_list: List[Dict], pruning_algorithm: str,
                 total_iteration: int, log_dir: str = '.', keep_intermediate_result: bool = False,
                 *args, **kwargs):
        new_api = ['evaluator', 'speedup', 'pruning_params']
        new_init_kwargs = {'evaluator': None, 'speedup': False, 'pruning_params': {}}
        old_api = ['finetuner', 'speedup', 'dummy_input', 'evaluator', 'pruning_params']
        old_init_kwargs = {'finetuner': None, 'evaluator': None, 'dummy_input': None, 'speedup': False, 'pruning_params': {}}
        init_kwargs = self._init_evaluator(model, new_api, new_init_kwargs, old_api, old_init_kwargs, args, kwargs)

        speedup = init_kwargs['speedup']
        pruning_params = init_kwargs['pruning_params']

        task_generator = LinearTaskGenerator(total_iteration=total_iteration,
                                             origin_model=model,
                                             origin_config_list=config_list,
                                             log_dir=log_dir,
                                             keep_intermediate_result=keep_intermediate_result)
        if 'traced_optimizer' in pruning_params:
            pruning_params['traced_optimizer'] = OptimizerConstructHelper.from_trace(model, pruning_params['traced_optimizer'])
        pruner = PRUNER_DICT[pruning_algorithm](None, None, **pruning_params)

        if self.using_evaluator:
            super().__init__(pruner, task_generator, evaluator=self.evaluator, speedup=speedup, reset_weight=False)
        else:
            super().__init__(pruner, task_generator, finetuner=self.finetuner, speedup=speedup, dummy_input=self.dummy_input,
                             evaluator=self._evaluator, reset_weight=False)  # type: ignore


class AGPPruner(IterativePruner):
    __doc__ = r"""
    This is an iterative pruner, which the sparsity is increased from an initial sparsity value :math:`s_{i}` (usually 0) to a final sparsity value :math:`s_{f}` over a span of :math:`n` pruning iterations,
    starting at training step :math:`t_{0}` and with pruning frequency :math:`\Delta t`:

    :math:`s_{t}=s_{f}+\left(s_{i}-s_{f}\right)\left(1-\frac{t-t_{0}}{n \Delta t}\right)^{3} \text { for } t \in\left\{t_{0}, t_{0}+\Delta t, \ldots, t_{0} + n \Delta t\right\}`
    """ + r"""

    For more details please refer to `To prune, or not to prune: exploring the efficacy of pruning for model compression <https://arxiv.org/abs/1710.01878>`__\.

    Parameters
    ----------
    model
        The origin unwrapped pytorch model to be pruned.
    config_list
        The origin config list provided by the user.
    pruning_algorithm
        Supported pruning algorithm ['level', 'l1', 'l2', 'fpgm', 'slim', 'apoz', 'mean_activation', 'taylorfo', 'admm'].
        This iterative pruner will use the chosen corresponding pruner to prune the model in each iteration.
    total_iteration
        The total iteration number.
    log_dir
        The log directory use to saving the result, you can find the best result under this folder.
    keep_intermediate_result
        If keeping the intermediate result, including intermediate model and masks during each iteration.
    evaluator
        ``evaluator`` is used to replace the previous ``finetuner``, ``dummy_input`` and old ``evaluator`` API.
        {evaluator_docstring}
        The old API (``finetuner``, ``dummy_input`` and old ``evaluator``) is still supported and will be deprecated in v3.0.
        If you want to consult the old API, please refer to `v2.8 pruner API <https://nni.readthedocs.io/en/v2.8/reference/compression/pruner.html>`__.
    speedup
        If set True, speedup the model at the end of each iteration to make the pruned model compact.
    pruning_params
        If the chosen pruning_algorithm has extra parameters, put them as a dict to pass in.

    Notes
    -----
    For detailed example please refer to :githublink:`examples/model_compress/pruning/iterative_pruning_torch.py <examples/model_compress/pruning/iterative_pruning_torch.py>`
    """.format(evaluator_docstring=_EVALUATOR_DOCSTRING)

    @overload
    def __init__(self, model: Module, config_list: List[Dict], pruning_algorithm: str,
                 total_iteration: int, log_dir: str = '.', keep_intermediate_result: bool = False,
                 evaluator: Evaluator | None = None, speedup: bool = False,
                 pruning_params: Dict = {}):
        ...

    @overload
    def __init__(self, model: Module, config_list: List[Dict], pruning_algorithm: str,
                 total_iteration: int, log_dir: str = '.', keep_intermediate_result: bool = False,
                 finetuner: _LEGACY_FINETUNER | None = None, speedup: bool = False, dummy_input: Any | None = None,
                 evaluator: _LEGACY_EVALUATOR | None = None, pruning_params: Dict = {}):
        ...

    def __init__(self, model: Module, config_list: List[Dict], pruning_algorithm: str,
                 total_iteration: int, log_dir: str = '.', keep_intermediate_result: bool = False,
                 *args, **kwargs):
        new_api = ['evaluator', 'speedup', 'pruning_params']
        new_init_kwargs = {'evaluator': None, 'speedup': False, 'pruning_params': {}}
        old_api = ['finetuner', 'speedup', 'dummy_input', 'evaluator', 'pruning_params']
        old_init_kwargs = {'finetuner': None, 'evaluator': None, 'dummy_input': None, 'speedup': False, 'pruning_params': {}}
        init_kwargs = self._init_evaluator(model, new_api, new_init_kwargs, old_api, old_init_kwargs, args, kwargs)

        speedup = init_kwargs['speedup']
        pruning_params = init_kwargs['pruning_params']

        task_generator = AGPTaskGenerator(total_iteration=total_iteration,
                                          origin_model=model,
                                          origin_config_list=config_list,
                                          log_dir=log_dir,
                                          keep_intermediate_result=keep_intermediate_result)
        if 'traced_optimizer' in pruning_params:
            pruning_params['traced_optimizer'] = OptimizerConstructHelper.from_trace(model, pruning_params['traced_optimizer'])
        pruner = PRUNER_DICT[pruning_algorithm](None, None, **pruning_params)

        if self.using_evaluator:
            super().__init__(pruner, task_generator, evaluator=self.evaluator, speedup=speedup, reset_weight=False)
        else:
            super().__init__(pruner, task_generator, finetuner=self.finetuner, speedup=speedup, dummy_input=self.dummy_input,
                             evaluator=self._evaluator, reset_weight=False)  # type: ignore


class LotteryTicketPruner(IterativePruner):
    __doc__ = r"""
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
    """ + r"""

    Parameters
    ----------
    model
        The origin unwrapped pytorch model to be pruned.
    config_list
        The origin config list provided by the user.
    pruning_algorithm
        Supported pruning algorithm ['level', 'l1', 'l2', 'fpgm', 'slim', 'apoz', 'mean_activation', 'taylorfo', 'admm'].
        This iterative pruner will use the chosen corresponding pruner to prune the model in each iteration.
    total_iteration
        The total iteration number.
    log_dir
        The log directory use to saving the result, you can find the best result under this folder.
    keep_intermediate_result
        If keeping the intermediate result, including intermediate model and masks during each iteration.
    evaluator
        ``evaluator`` is used to replace the previous ``finetuner``, ``dummy_input`` and old ``evaluator`` API.
        {evaluator_docstring}
        The old API (``finetuner``, ``dummy_input`` and old ``evaluator``) is still supported and will be deprecated in v3.0.
        If you want to consult the old API, please refer to `v2.8 pruner API <https://nni.readthedocs.io/en/v2.8/reference/compression/pruner.html>`__.
    speedup
        If set True, speedup the model at the end of each iteration to make the pruned model compact.
    reset_weight
        If set True, the model weight will reset to the original model weight at the end of each iteration step.
    pruning_params
        If the chosen pruning_algorithm has extra parameters, put them as a dict to pass in.

    Notes
    -----
    For detailed example please refer to :githublink:`examples/model_compress/pruning/iterative_pruning_torch.py <examples/model_compress/pruning/iterative_pruning_torch.py>`
    """.format(evaluator_docstring=_EVALUATOR_DOCSTRING)

    @overload
    def __init__(self, model: Module, config_list: List[Dict], pruning_algorithm: str,
                 total_iteration: int, log_dir: str = '.', keep_intermediate_result: bool = False,
                 evaluator: Evaluator | None = None, speedup: bool = False,
                 reset_weight: bool = True, pruning_params: Dict = {}):
        ...

    @overload
    def __init__(self, model: Module, config_list: List[Dict], pruning_algorithm: str,
                 total_iteration: int, log_dir: str = '.', keep_intermediate_result: bool = False,
                 finetuner: _LEGACY_FINETUNER | None = None, speedup: bool = False, dummy_input: Optional[Tensor] = None,
                 evaluator: _LEGACY_EVALUATOR | None = None, reset_weight: bool = True,
                 pruning_params: Dict = {}):
        ...

    def __init__(self, model: Module, config_list: List[Dict], pruning_algorithm: str,
                 total_iteration: int, log_dir: str = '.', keep_intermediate_result: bool = False,
                 *args, **kwargs):
        new_api = ['evaluator', 'speedup', 'reset_weight', 'pruning_params']
        new_init_kwargs = {'evaluator': None, 'speedup': False, 'reset_weight': True, 'pruning_params': {}}
        old_api = ['finetuner', 'speedup', 'dummy_input', 'evaluator', 'reset_weight', 'pruning_params']
        old_init_kwargs = {'finetuner': None, 'evaluator': None, 'dummy_input': None, 'speedup': False,
                           'reset_weight': True, 'pruning_params': {}}
        init_kwargs = self._init_evaluator(model, new_api, new_init_kwargs, old_api, old_init_kwargs, args, kwargs)

        speedup = init_kwargs['speedup']
        reset_weight = init_kwargs['reset_weight']
        pruning_params = init_kwargs['pruning_params']

        task_generator = LotteryTicketTaskGenerator(total_iteration=total_iteration,
                                                    origin_model=model,
                                                    origin_config_list=config_list,
                                                    log_dir=log_dir,
                                                    keep_intermediate_result=keep_intermediate_result)
        if 'traced_optimizer' in pruning_params:
            pruning_params['traced_optimizer'] = OptimizerConstructHelper.from_trace(model, pruning_params['traced_optimizer'])
        pruner = PRUNER_DICT[pruning_algorithm](None, None, **pruning_params)

        if self.using_evaluator:
            super().__init__(pruner, task_generator, evaluator=self.evaluator, speedup=speedup, reset_weight=reset_weight)
        else:
            super().__init__(pruner, task_generator, finetuner=self.finetuner, speedup=speedup, dummy_input=self.dummy_input,
                             evaluator=self._evaluator, reset_weight=reset_weight)  # type: ignore


class SimulatedAnnealingPruner(IterativePruner):
    __doc__ = r"""
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
    model
        The origin unwrapped pytorch model to be pruned.
    config_list
        The origin config list provided by the user.
    evaluator
        ``evaluator`` is used to replace the previous ``finetuner``, ``dummy_input`` and old ``evaluator`` API.
        {evaluator_docstring}
        The old API (``finetuner``, ``dummy_input`` and old ``evaluator``) is still supported and will be deprecated in v3.0.
        If you want to consult the old API, please refer to `v2.8 pruner API <https://nni.readthedocs.io/en/v2.8/reference/compression/pruner.html>`__.
    start_temperature
        Start temperature of the simulated annealing process.
    stop_temperature
        Stop temperature of the simulated annealing process.
    cool_down_rate
        Cool down rate of the temperature.
    perturbation_magnitude
        Initial perturbation magnitude to the sparsities. The magnitude decreases with current temperature.
    pruning_algorithm
        Supported pruning algorithm ['level', 'l1', 'l2', 'fpgm', 'slim', 'apoz', 'mean_activation', 'taylorfo', 'admm'].
        This iterative pruner will use the chosen corresponding pruner to prune the model in each iteration.
    pruning_params
        If the chosen pruning_algorithm has extra parameters, put them as a dict to pass in.
    log_dir
        The log directory use to saving the result, you can find the best result under this folder.
    keep_intermediate_result
        If keeping the intermediate result, including intermediate model and masks during each iteration.
    speedup
        If set True, speedup the model at the end of each iteration to make the pruned model compact.

    Notes
    -----
    For detailed example please refer to :githublink:`examples/model_compress/pruning/simulated_anealing_pruning_torch.py <examples/model_compress/pruning/simulated_anealing_pruning_torch.py>`
    """.format(evaluator_docstring=_EVALUATOR_DOCSTRING)

    @overload
    def __init__(self, model: Module, config_list: List[Dict], evaluator: Evaluator,
                 start_temperature: float = 100, stop_temperature: float = 20, cool_down_rate: float = 0.9,
                 perturbation_magnitude: float = 0.35, pruning_algorithm: str = 'level', pruning_params: Dict = {},
                 log_dir: Union[str, Path] = '.', keep_intermediate_result: bool = False, speedup: bool = False):
        ...

    @overload
    def __init__(self, model: Module, config_list: List[Dict], evaluator: _LEGACY_EVALUATOR,
                 start_temperature: float = 100, stop_temperature: float = 20, cool_down_rate: float = 0.9,
                 perturbation_magnitude: float = 0.35, pruning_algorithm: str = 'level', pruning_params: Dict = {},
                 log_dir: Union[str, Path] = '.', keep_intermediate_result: bool = False,
                 finetuner: _LEGACY_FINETUNER | None = None, speedup: bool = False,
                 dummy_input: Optional[Tensor] = None):
        ...

    def __init__(self, model: Module, config_list: List[Dict], *args, **kwargs):
        new_api = ['evaluator', 'start_temperature', 'stop_temperature', 'cool_down_rate', 'perturbation_magnitude',
                   'pruning_algorithm', 'pruning_params', 'log_dir', 'keep_intermediate_result', 'speedup']
        new_init_kwargs = {'start_temperature': 100, 'stop_temperature': 20, 'cool_down_rate': 0.9,
                           'perturbation_magnitude': 0.35, 'pruning_algorithm': 'level', 'pruning_params': {},
                           'log_dir': '.', 'keep_intermediate_result': False, 'speedup': False}
        old_api = ['evaluator', 'start_temperature', 'stop_temperature', 'cool_down_rate', 'perturbation_magnitude',
                   'pruning_algorithm', 'pruning_params', 'log_dir', 'keep_intermediate_result', 'finetuner',
                   'speedup', 'dummy_input']
        old_init_kwargs = {'start_temperature': 100, 'stop_temperature': 20, 'cool_down_rate': 0.9,
                           'perturbation_magnitude': 0.35, 'pruning_algorithm': 'level', 'pruning_params': {},
                           'log_dir': '.', 'keep_intermediate_result': False, 'finetuner': None, 'speedup': False,
                           'dummy_input': None}
        init_kwargs = self._init_evaluator(model, new_api, new_init_kwargs, old_api, old_init_kwargs, args, kwargs)

        start_temperature = init_kwargs['start_temperature']
        stop_temperature = init_kwargs['stop_temperature']
        cool_down_rate = init_kwargs['cool_down_rate']
        perturbation_magnitude = init_kwargs['perturbation_magnitude']
        pruning_algorithm = init_kwargs['pruning_algorithm']
        pruning_params = init_kwargs['pruning_params']
        log_dir = init_kwargs['log_dir']
        keep_intermediate_result = init_kwargs['keep_intermediate_result']
        speedup = init_kwargs['speedup']

        task_generator = SimulatedAnnealingTaskGenerator(origin_model=model,
                                                         origin_config_list=config_list,
                                                         start_temperature=start_temperature,
                                                         stop_temperature=stop_temperature,
                                                         cool_down_rate=cool_down_rate,
                                                         perturbation_magnitude=perturbation_magnitude,
                                                         log_dir=log_dir,
                                                         keep_intermediate_result=keep_intermediate_result)
        if 'traced_optimizer' in pruning_params:
            pruning_params['traced_optimizer'] = \
                OptimizerConstructHelper.from_trace(model, pruning_params['traced_optimizer'])  # type: ignore
        pruner = PRUNER_DICT[pruning_algorithm](None, None, **pruning_params)

        if self.using_evaluator:
            super().__init__(pruner, task_generator, evaluator=self.evaluator, speedup=speedup, reset_weight=False)
        else:
            super().__init__(pruner, task_generator, finetuner=self.finetuner, speedup=speedup,
                             dummy_input=self.dummy_input, evaluator=self._evaluator, reset_weight=False)  # type: ignore
