# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging

import warnings
from threading import Thread
from typing import Any, List, Union, cast

import colorama

import torch
import torch.nn as nn
from nni.experiment import Experiment, RunMode
from nni.experiment.config.training_services import RemoteConfig

from .config import (
    RetiariiExeConfig, OneshotEngineConfig, BaseEngineConfig,
    PyEngineConfig, CgoEngineConfig, BenchmarkEngineConfig
)
from ..codegen import model_to_pytorch_script
from ..converter import convert_to_graph
from ..converter.graph_gen import GraphConverterWithShape
from ..execution import list_models, set_execution_engine
from ..execution.utils import get_mutation_dict
from ..graph import Evaluator
from ..integration import RetiariiAdvisor
from ..mutator import Mutator
from ..nn.pytorch.mutator import (
    extract_mutation_from_pt_module, process_inline_mutation, process_evaluator_mutations, process_oneshot_mutations
)
from ..oneshot.interface import BaseOneShotTrainer
from ..serializer import is_model_wrapped
from ..strategy import BaseStrategy
from ..strategy.utils import dry_run_for_formatted_search_space

_logger = logging.getLogger(__name__)


__all__ = ['RetiariiExperiment']


def preprocess_model(base_model, evaluator, applied_mutators, full_ir=True, dummy_input=None, oneshot=False):
    # TODO: this logic might need to be refactored into execution engine
    if oneshot:
        base_model_ir, mutators = process_oneshot_mutations(base_model, evaluator)
    elif full_ir:
        try:
            script_module = torch.jit.script(base_model)
        except Exception as e:
            _logger.error('Your base model cannot be parsed by torch.jit.script, please fix the following error:')
            raise e
        if dummy_input is not None:
            # FIXME: this is a workaround as full tensor is not supported in configs
            dummy_input = torch.randn(*dummy_input)
            converter = GraphConverterWithShape()
            base_model_ir = convert_to_graph(script_module, base_model, converter, dummy_input=dummy_input)
        else:
            base_model_ir = convert_to_graph(script_module, base_model)
        # handle inline mutations
        mutators = process_inline_mutation(base_model_ir)
    else:
        base_model_ir, mutators = extract_mutation_from_pt_module(base_model)
    base_model_ir.evaluator = evaluator

    if mutators is not None and applied_mutators:
        raise RuntimeError('Have not supported mixed usage of LayerChoice/InputChoice and mutators, '
                           'do not use mutators when you use LayerChoice/InputChoice')
    if mutators is not None:
        applied_mutators = mutators

    # Add mutations on evaluators
    applied_mutators += process_evaluator_mutations(evaluator, applied_mutators)

    return base_model_ir, applied_mutators


def debug_mutated_model(base_model, evaluator, applied_mutators):
    """
    Locally run only one trial without launching an experiment for debug purpose, then exit.
    For example, it can be used to quickly check shape mismatch.

    Specifically, it applies mutators (default to choose the first candidate for the choices)
    to generate a new model, then run this model locally.

    The model will be parsed with graph execution engine.

    Parameters
    ----------
    base_model : nni.retiarii.nn.pytorch.nn.Module
        the base model
    evaluator : nni.retiarii.graph.Evaluator
        the training class of the generated models
    applied_mutators : list
        a list of mutators that will be applied on the base model for generating a new model
    """
    base_model_ir, applied_mutators = preprocess_model(base_model, evaluator, applied_mutators)
    from ..strategy import _LocalDebugStrategy
    strategy = _LocalDebugStrategy()
    strategy.run(base_model_ir, applied_mutators)
    _logger.info('local debug completed!')


class RetiariiExperiment(Experiment):
    """
    The entry for a NAS experiment.
    Users can use this class to start/stop or inspect an experiment, like exporting the results.

    Experiment is a sub-class of :class:`nni.experiment.Experiment`, there are many similarities such as
    configurable training service to distributed running the experiment on remote server.
    But unlike :class:`nni.experiment.Experiment`, RetiariiExperiment doesn't support configure:

    - ``trial_code_directory``, which can only be current working directory.
    - ``search_space``, which is auto-generated in NAS.
    - ``trial_command``, which must be ``python -m nni.retiarii.trial_entry`` to launch the modulized trial code.

    RetiariiExperiment also doesn't have tuner/assessor/advisor, because they are also implemented in strategy.

    Also, unlike :class:`nni.experiment.Experiment` which is bounded to a node server,
    RetiariiExperiment optionally starts a node server to schedule the trials, when the strategy is a multi-trial strategy.
    When the strategy is one-shot, the step of launching node server is omitted, and the experiment is run locally by default.

    Configurations of experiments, such as execution engine, number of GPUs allocated,
    should be put into a :class:`RetiariiExeConfig` and used as an argument of :meth:`RetiariiExperiment.run`.

    Parameters
    ----------
    base_model : nn.Module
        The model defining the search space / base skeleton without mutation.
        It should be wrapped by decorator ``nni.retiarii.model_wrapper``.
    evaluator : nni.retiarii.Evaluator, default = None
        Evaluator for the experiment.
        If you are using a one-shot trainer, it should be placed here, although this usage is deprecated.
    applied_mutators : list of nni.retiarii.Mutator, default = None
        Mutators os mutate the base model. If none, mutators are skipped.
        Note that when ``base_model`` uses inline mutations (e.g., LayerChoice), ``applied_mutators`` must be empty / none.
    strategy : nni.retiarii.strategy.BaseStrategy, default = None
        Exploration strategy. Can be multi-trial or one-shot.
    trainer : BaseOneShotTrainer
        Kept for compatibility purposes.

    Examples
    --------
    Multi-trial NAS:

    >>> base_model = Net()
    >>> search_strategy = strategy.Random()
    >>> model_evaluator = FunctionalEvaluator(evaluate_model)
    >>> exp = RetiariiExperiment(base_model, model_evaluator, [], search_strategy)
    >>> exp_config = RetiariiExeConfig('local')
    >>> exp_config.trial_concurrency = 2
    >>> exp_config.max_trial_number = 20
    >>> exp_config.training_service.use_active_gpu = False
    >>> exp.run(exp_config, 8081)

    One-shot NAS:

    >>> base_model = Net()
    >>> search_strategy = strategy.DARTS()
    >>> evaluator = pl.Classification(train_dataloader=train_loader, val_dataloaders=valid_loader)
    >>> exp = RetiariiExperiment(base_model, evaluator, [], search_strategy)
    >>> exp_config = RetiariiExeConfig()
    >>> exp_config.execution_engine = 'oneshot'  # must be set of one-shot strategy
    >>> exp.run(exp_config)

    Export top models:

    >>> for model_dict in exp.export_top_models(formatter='dict'):
    ...     print(model_dict)
    >>> with nni.retarii.fixed_arch(model_dict):
    ...     final_model = Net()
    """

    def __init__(self, base_model: nn.Module,
                 evaluator: Union[BaseOneShotTrainer, Evaluator] = cast(Evaluator, None),
                 applied_mutators: List[Mutator] = cast(List[Mutator], None),
                 strategy: BaseStrategy = cast(BaseStrategy, None),
                 trainer: BaseOneShotTrainer = cast(BaseOneShotTrainer, None)):
        super().__init__(None)
        self.config: RetiariiExeConfig = cast(RetiariiExeConfig, None)

        if trainer is not None:
            warnings.warn('Usage of `trainer` in RetiariiExperiment is deprecated and will be removed soon. '
                          'Please consider specifying it as a positional argument, or use `evaluator`.', DeprecationWarning)
            evaluator = trainer

        if evaluator is None:
            raise ValueError('Evaluator should not be none.')

        self.base_model = base_model
        self.evaluator: Union[Evaluator, BaseOneShotTrainer] = evaluator
        self.applied_mutators = applied_mutators
        self.strategy = strategy

        self._dispatcher = None
        self._dispatcher_thread = None

        # check for sanity
        if not is_model_wrapped(base_model):
            warnings.warn(colorama.Style.BRIGHT + colorama.Fore.RED +
                          '`@model_wrapper` is missing for the base model. The experiment might still be able to run, '
                          'but it may cause inconsistent behavior compared to the time when you add it.' + colorama.Style.RESET_ALL,
                          RuntimeWarning)

    def _run_strategy(self, config: RetiariiExeConfig):
        base_model_ir, self.applied_mutators = preprocess_model(
            self.base_model, self.evaluator, self.applied_mutators,
            full_ir=not isinstance(config.execution_engine, (PyEngineConfig, BenchmarkEngineConfig)),
            dummy_input=config.execution_engine.dummy_input
                if isinstance(config.execution_engine, (BaseEngineConfig, CgoEngineConfig)) else None
        )

        _logger.info('Start strategy...')
        search_space = dry_run_for_formatted_search_space(base_model_ir, self.applied_mutators)
        self.update_search_space(search_space)
        self.strategy.run(base_model_ir, self.applied_mutators)
        _logger.info('Strategy exit')
        # TODO: find out a proper way to show no more trial message on WebUI

    def _create_execution_engine(self, config: RetiariiExeConfig) -> None:
        #TODO: we will probably need a execution engine factory to make this clean and elegant
        if isinstance(config.execution_engine, BaseEngineConfig):
            from ..execution.base import BaseExecutionEngine
            engine = BaseExecutionEngine(self.port, self.url_prefix)
        elif isinstance(config.execution_engine, CgoEngineConfig):
            from ..execution.cgo_engine import CGOExecutionEngine

            assert not isinstance(config.training_service, list) \
                and config.training_service.platform == 'remote', \
                "CGO execution engine currently only supports remote training service"
            assert config.execution_engine.batch_waiting_time is not None \
                and config.execution_engine.max_concurrency_cgo is not None
            engine = CGOExecutionEngine(cast(RemoteConfig, config.training_service),
                                        max_concurrency=config.execution_engine.max_concurrency_cgo,
                                        batch_waiting_time=config.execution_engine.batch_waiting_time,
                                        rest_port=self.port,
                                        rest_url_prefix=self.url_prefix)
        elif isinstance(config.execution_engine, PyEngineConfig):
            from ..execution.python import PurePythonExecutionEngine
            engine = PurePythonExecutionEngine(self.port, self.url_prefix)
        elif isinstance(config.execution_engine, BenchmarkEngineConfig):
            from ..execution.benchmark import BenchmarkExecutionEngine
            assert config.execution_engine.benchmark is not None, \
                '"benchmark" must be set when benchmark execution engine is used.'
            engine = BenchmarkExecutionEngine(config.execution_engine.benchmark)
        else:
            raise ValueError(f'Unsupported engine type: {config.execution_engine}')
        set_execution_engine(engine)

    def start(self, *args, **kwargs) -> None:
        """
        By design, the only different between `start` and `run` is that `start` is asynchronous,
        while `run` waits the experiment to complete. RetiariiExperiment always waits the experiment
        to complete as strategy runs in foreground.
        """
        raise NotImplementedError('RetiariiExperiment is not supposed to provide `start` method')

    def run(self,
            config: RetiariiExeConfig | None = None,
            port: int = 8080,
            debug: bool = False) -> None:
        """
        Run the experiment.
        This function will block until experiment finish or error.
        """
        if isinstance(self.evaluator, BaseOneShotTrainer):
            # TODO: will throw a deprecation warning soon
            # warnings.warn('You are using the old implementation of one-shot algos based on One-shot trainer. '
            #               'We will try to convert this trainer to our new implementation to run the algorithm. '
            #               'In case you want to stick to the old implementation, '
            #               'please consider using ``trainer.fit()`` instead of experiment.', DeprecationWarning)
            self.evaluator.fit()
            return

        if config is None:
            warnings.warn('config = None is deprecate in future. If you are running a one-shot experiment, '
                          'please consider creating a config and set execution engine to `oneshot`.', DeprecationWarning)
            self.config = RetiariiExeConfig()
            self.config.execution_engine = OneshotEngineConfig()
        else:
            self.config = config

        if isinstance(self.config.execution_engine, OneshotEngineConfig) \
            or (isinstance(self.config.execution_engine, str) and self.config.execution_engine == 'oneshot'):
            # this is hacky, will be refactored when oneshot can run on training services
            base_model_ir, self.applied_mutators = preprocess_model(self.base_model, self.evaluator, self.applied_mutators, oneshot=True)
            self.strategy.run(base_model_ir, self.applied_mutators)
        else:
            ws_url = f'ws://localhost:{port}/tuner'
            canonicalized_config = self._start_impl(port, debug, RunMode.Background, ws_url, ['retiarii'])
            canonicalized_config = cast(RetiariiExeConfig, canonicalized_config)
            self._dispatcher = RetiariiAdvisor(ws_url)
            self._dispatcher_thread = Thread(target=self._dispatcher.run, daemon=True)
            self._dispatcher_thread.start()
            # FIXME: engine cannot be created twice
            self._create_execution_engine(canonicalized_config)
            try:
                self._run_strategy(canonicalized_config)
                # FIXME: move this logic to strategy with a new API provided by execution engine
                self._wait_completion()
            except KeyboardInterrupt:
                _logger.warning('KeyboardInterrupt detected')
                self.stop()
            _logger.info('Search process is done, the experiment is still alive, `stop()` can terminate the experiment.')

    def stop(self) -> None:
        """
        Stop background experiment.
        """
        _logger.info('Stopping experiment, please wait...')
        self._stop_impl()
        if self._dispatcher_thread:
            self._dispatcher_thread.join()
        self._dispatcher = cast(RetiariiAdvisor, None)
        self._dispatcher_thread = None
        _logger.info('Experiment stopped')

    def export_top_models(self, top_k: int = 1, optimize_mode: str = 'maximize', formatter: str = 'dict') -> Any:
        """
        Export several top performing models.

        For one-shot algorithms, only top-1 is supported. For others, ``optimize_mode`` and ``formatter`` are
        available for customization.

        The concrete behavior of export depends on each strategy.
        See the documentation of each strategy for detailed specifications.

        Parameters
        ----------
        top_k : int
            How many models are intended to be exported.
        optimize_mode : str
            ``maximize`` or ``minimize``. Not supported by one-shot algorithms.
            ``optimize_mode`` is likely to be removed and defined in strategy in future.
        formatter : str
            Support ``code`` and ``dict``. Not supported by one-shot algorithms.
            If ``code``, the python code of model will be returned.
            If ``dict``, the mutation history will be returned.
        """
        # TODO: the base class may also need this method
        if formatter == 'code':
            config = self.config.canonical_copy()
            assert not isinstance(config.execution_engine, PyEngineConfig), \
                'You should use `dict` formatter when using Python execution engine.'
        if isinstance(self.evaluator, BaseOneShotTrainer):
            assert top_k == 1, 'Only support top_k is 1 for now.'
            return self.evaluator.export()
        try:
            # this currently works for one-shot algorithms
            return self.strategy.export_top_models(top_k=top_k)
        except NotImplementedError:
            # when strategy hasn't implemented its own export logic
            all_models = filter(lambda m: m.metric is not None, list_models())
            assert optimize_mode in ['maximize', 'minimize']
            all_models = sorted(all_models, key=lambda m: cast(float, m.metric), reverse=optimize_mode == 'maximize')
            assert formatter in ['code', 'dict'], 'Export formatter other than "code" and "dict" is not supported yet.'
            if formatter == 'code':
                return [model_to_pytorch_script(model) for model in all_models[:top_k]]
            elif formatter == 'dict':
                return [get_mutation_dict(model) for model in all_models[:top_k]]
