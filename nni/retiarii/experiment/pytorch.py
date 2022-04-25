# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import atexit
import logging
import os
import socket
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from subprocess import Popen
from threading import Thread
from typing import Any, List, Optional, Union, cast

import colorama
import psutil
import torch
import torch.nn as nn
import nni.runtime.log
from nni.common.device import GPUDevice
from nni.experiment import Experiment, launcher, management, rest
from nni.experiment.config import utils
from nni.experiment.config.base import ConfigBase
from nni.experiment.config.training_service import TrainingServiceConfig
from nni.experiment.config.training_services import RemoteConfig
from nni.experiment.pipe import Pipe
from nni.tools.nnictl.command_utils import kill_command

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


__all__ = ['RetiariiExeConfig', 'RetiariiExperiment']


@dataclass(init=False)
class RetiariiExeConfig(ConfigBase):
    experiment_name: Optional[str] = None
    search_space: Any = ''  # TODO: remove
    trial_command: str = '_reserved'
    trial_code_directory: utils.PathLike = '.'
    trial_concurrency: int
    trial_gpu_number: int = 0
    devices: Optional[List[Union[str, GPUDevice]]] = None
    max_experiment_duration: Optional[str] = None
    max_trial_number: Optional[int] = None
    max_concurrency_cgo: Optional[int] = None
    batch_waiting_time: Optional[int] = None
    nni_manager_ip: Optional[str] = None
    debug: bool = False
    log_level: Optional[str] = None
    experiment_working_directory: utils.PathLike = '~/nni-experiments'
    # remove configuration of tuner/assessor/advisor
    training_service: TrainingServiceConfig
    execution_engine: str = 'py'

    # input used in GraphConverterWithShape. Currently support shape tuple only.
    dummy_input: Optional[List[int]] = None

    # input used for benchmark engine.
    benchmark: Optional[str] = None

    def __init__(self, training_service_platform: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if training_service_platform is not None:
            assert 'training_service' not in kwargs
            self.training_service = utils.training_service_config_factory(platform=training_service_platform)
        self.__dict__['trial_command'] = 'python3 -m nni.retiarii.trial_entry py'

    def __setattr__(self, key, value):
        fixed_attrs = {'search_space': '',
                       'trial_command': '_reserved'}
        if key in fixed_attrs and fixed_attrs[key] != value:
            raise AttributeError(f'{key} is not supposed to be set in Retiarii mode by users!')
        # 'trial_code_directory' is handled differently because the path will be converted to absolute path by us
        if key == 'trial_code_directory' and not (str(value) == '.' or os.path.isabs(value)):
            raise AttributeError(f'{key} is not supposed to be set in Retiarii mode by users!')
        if key == 'execution_engine':
            assert value in ['base', 'py', 'cgo', 'benchmark', 'oneshot'], f'The specified execution engine "{value}" is not supported.'
            self.__dict__['trial_command'] = 'python3 -m nni.retiarii.trial_entry ' + value
        self.__dict__[key] = value

    def validate(self, initialized_tuner: bool = False) -> None:
        super().validate()

    @property
    def _canonical_rules(self):
        return _canonical_rules

    @property
    def _validation_rules(self):
        return _validation_rules


_canonical_rules = {
}

_validation_rules = {
    'trial_code_directory': lambda value: (Path(value).is_dir(), f'"{value}" does not exist or is not directory'),
    'trial_concurrency': lambda value: value > 0,
    'trial_gpu_number': lambda value: value >= 0,
    'max_trial_number': lambda value: value > 0,
    'log_level': lambda value: value in ["trace", "debug", "info", "warning", "error", "fatal"],
    'training_service': lambda value: (type(value) is not TrainingServiceConfig, 'cannot be abstract base class')
}


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

    def __init__(self, base_model: nn.Module, evaluator: Union[BaseOneShotTrainer, Evaluator] = cast(Evaluator, None),
                 applied_mutators: List[Mutator] = cast(List[Mutator], None), strategy: BaseStrategy = cast(BaseStrategy, None),
                 trainer: BaseOneShotTrainer = cast(BaseOneShotTrainer, None)):
        if trainer is not None:
            warnings.warn('Usage of `trainer` in RetiariiExperiment is deprecated and will be removed soon. '
                          'Please consider specifying it as a positional argument, or use `evaluator`.', DeprecationWarning)
            evaluator = trainer

        if evaluator is None:
            raise ValueError('Evaluator should not be none.')

        # TODO: The current design of init interface of Retiarii experiment needs to be reviewed.
        self.config: RetiariiExeConfig = cast(RetiariiExeConfig, None)
        self.port: Optional[int] = None

        self.base_model = base_model
        self.evaluator: Union[Evaluator, BaseOneShotTrainer] = evaluator
        self.applied_mutators = applied_mutators
        self.strategy = strategy

        from nni.retiarii.oneshot.pytorch.strategy import OneShotStrategy
        if not isinstance(strategy, OneShotStrategy):
            self._dispatcher = RetiariiAdvisor()
        else:
            self._dispatcher = cast(RetiariiAdvisor, None)
        self._dispatcher_thread: Optional[Thread] = None
        self._proc: Optional[Popen] = None
        self._pipe: Optional[Pipe] = None

        self.url_prefix = None

        # check for sanity
        if not is_model_wrapped(base_model):
            warnings.warn(colorama.Style.BRIGHT + colorama.Fore.RED +
                          '`@model_wrapper` is missing for the base model. The experiment might still be able to run, '
                          'but it may cause inconsistent behavior compared to the time when you add it.' + colorama.Style.RESET_ALL,
                          RuntimeWarning)

    def _start_strategy(self):
        base_model_ir, self.applied_mutators = preprocess_model(
            self.base_model, self.evaluator, self.applied_mutators,
            full_ir=self.config.execution_engine not in ['py', 'benchmark'],
            dummy_input=self.config.dummy_input
        )

        _logger.info('Start strategy...')
        search_space = dry_run_for_formatted_search_space(base_model_ir, self.applied_mutators)
        self.update_search_space(search_space)
        self.strategy.run(base_model_ir, self.applied_mutators)
        _logger.info('Strategy exit')
        # TODO: find out a proper way to show no more trial message on WebUI
        # self._dispatcher.mark_experiment_as_ending()

    def start(self, port: int = 8080, debug: bool = False) -> None:
        """
        Start the experiment in background.
        This method will raise exception on failure.
        If it returns, the experiment should have been successfully started.
        Parameters
        ----------
        port
            The port of web UI.
        debug
            Whether to start in debug mode.
        """
        atexit.register(self.stop)

        self.config = self.config.canonical_copy()

        # we will probably need a execution engine factory to make this clean and elegant
        if self.config.execution_engine == 'base':
            from ..execution.base import BaseExecutionEngine
            engine = BaseExecutionEngine()
        elif self.config.execution_engine == 'cgo':
            from ..execution.cgo_engine import CGOExecutionEngine

            assert self.config.training_service.platform == 'remote', \
                "CGO execution engine currently only supports remote training service"
            assert self.config.batch_waiting_time is not None and self.config.max_concurrency_cgo is not None
            devices = self._construct_devices()
            engine = CGOExecutionEngine(devices,
                                        max_concurrency=self.config.max_concurrency_cgo,
                                        batch_waiting_time=self.config.batch_waiting_time)
        elif self.config.execution_engine == 'py':
            from ..execution.python import PurePythonExecutionEngine
            engine = PurePythonExecutionEngine()
        elif self.config.execution_engine == 'benchmark':
            from ..execution.benchmark import BenchmarkExecutionEngine
            assert self.config.benchmark is not None, '"benchmark" must be set when benchmark execution engine is used.'
            engine = BenchmarkExecutionEngine(self.config.benchmark)
        else:
            raise ValueError(f'Unsupported engine type: {self.config.execution_engine}')
        set_execution_engine(engine)

        self.id = management.generate_experiment_id()

        if self.config.experiment_working_directory is not None:
            log_dir = Path(self.config.experiment_working_directory, self.id, 'log')
        else:
            log_dir = Path.home() / f'nni-experiments/{self.id}/log'
        nni.runtime.log.start_experiment_log(self.id, log_dir, debug)

        self._proc, self._pipe = launcher.start_experiment_retiarii(self.id, self.config, port, debug)
        assert self._proc is not None
        assert self._pipe is not None

        self.port = port  # port will be None if start up failed

        # dispatcher must be launched after pipe initialized
        # the logic to launch dispatcher in background should be refactored into dispatcher api
        self._dispatcher = self._create_dispatcher()
        self._dispatcher_thread = Thread(target=self._dispatcher.run)
        self._dispatcher_thread.start()

        ips = [self.config.nni_manager_ip]
        for interfaces in psutil.net_if_addrs().values():
            for interface in interfaces:
                if interface.family == socket.AF_INET:
                    ips.append(interface.address)
        ips = [f'http://{ip}:{port}' for ip in ips if ip]
        msg = 'Web UI URLs: ' + colorama.Fore.CYAN + ' '.join(ips) + colorama.Style.RESET_ALL
        _logger.info(msg)

        exp_status_checker = Thread(target=self._check_exp_status)
        exp_status_checker.start()
        self._start_strategy()
        # TODO: the experiment should be completed, when strategy exits and there is no running job
        _logger.info('Waiting for experiment to become DONE (you can ctrl+c if there is no running trial jobs)...')
        exp_status_checker.join()

    def _construct_devices(self):
        devices = []
        if hasattr(self.config.training_service, 'machine_list'):
            for machine in cast(RemoteConfig, self.config.training_service).machine_list:
                assert machine.gpu_indices is not None, \
                    'gpu_indices must be set in RemoteMachineConfig for CGO execution engine'
                assert isinstance(machine.gpu_indices, list), 'gpu_indices must be a list'
                for gpu_idx in machine.gpu_indices:
                    devices.append(GPUDevice(machine.host, gpu_idx))
        return devices

    def _create_dispatcher(self):
        return self._dispatcher

    def run(self, config: Optional[RetiariiExeConfig] = None, port: int = 8080, debug: bool = False) -> None:
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

        if config is None:
            warnings.warn('config = None is deprecate in future. If you are running a one-shot experiment, '
                          'please consider creating a config and set execution engine to `oneshot`.', DeprecationWarning)
            config = RetiariiExeConfig()
            config.execution_engine = 'oneshot'

        if config.execution_engine == 'oneshot':
            base_model_ir, self.applied_mutators = preprocess_model(self.base_model, self.evaluator, self.applied_mutators, oneshot=True)
            self.strategy.run(base_model_ir, self.applied_mutators)
        else:
            assert config is not None, 'You are using classic search mode, config cannot be None!'
            self.config = config
            self.start(port, debug)

    def _check_exp_status(self) -> bool:
        """
        Run the experiment.
        This function will block until experiment finish or error.
        Return `True` when experiment done; or return `False` when experiment failed.
        """
        assert self._proc is not None
        try:
            while True:
                time.sleep(10)
                # this if is to deal with the situation that
                # nnimanager is cleaned up by ctrl+c first
                if self._proc.poll() is None:
                    status = self.get_status()
                else:
                    return False
                if status == 'DONE' or status == 'STOPPED':
                    return True
                if status == 'ERROR':
                    return False
        except KeyboardInterrupt:
            _logger.warning('KeyboardInterrupt detected')
        finally:
            self.stop()
        raise RuntimeError('Check experiment status failed.')

    def stop(self) -> None:
        """
        Stop background experiment.
        """
        _logger.info('Stopping experiment, please wait...')
        atexit.unregister(self.stop)

        # stop strategy first
        if self._dispatcher_thread is not None:
            self._dispatcher.stopping = True
            self._dispatcher_thread.join(timeout=1)

        if self.id is not None:
            nni.runtime.log.stop_experiment_log(self.id)
        if self._proc is not None:
            try:
                # this if is to deal with the situation that
                # nnimanager is cleaned up by ctrl+c first
                if self._proc.poll() is None:
                    rest.delete(self.port, '/experiment')
            except Exception as e:
                _logger.exception(e)
                _logger.warning('Cannot gracefully stop experiment, killing NNI process...')
                kill_command(self._proc.pid)

        if self._pipe is not None:
            self._pipe.close()

        self.id = cast(str, None)
        self.port = cast(int, None)
        self._proc = None
        self._pipe = None
        self._dispatcher = cast(RetiariiAdvisor, None)
        self._dispatcher_thread = None
        _logger.info('Experiment stopped')

    def export_top_models(self, top_k: int = 1, optimize_mode: str = 'maximize', formatter: str = 'dict') -> Any:
        """
        Export several top performing models.

        For one-shot algorithms, only top-1 is supported. For others, ``optimize_mode`` and ``formatter`` are
        available for customization.

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
        if formatter == 'code':
            assert self.config.execution_engine != 'py', 'You should use `dict` formatter when using Python execution engine.'
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
            all_models = sorted(all_models, key=lambda m: m.metric, reverse=optimize_mode == 'maximize')
            assert formatter in ['code', 'dict'], 'Export formatter other than "code" and "dict" is not supported yet.'
            if formatter == 'code':
                return [model_to_pytorch_script(model) for model in all_models[:top_k]]
            elif formatter == 'dict':
                return [get_mutation_dict(model) for model in all_models[:top_k]]

    def retrain_model(self, model):
        """
        this function retrains the exported model, and test it to output test accuracy
        """
        raise NotImplementedError
