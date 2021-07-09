# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import atexit
import logging
import time
from dataclasses import dataclass
import os
from pathlib import Path
import socket
from subprocess import Popen
from threading import Thread
import time
from typing import Any, List, Optional, Union

import colorama
import psutil

import torch
import torch.nn as nn
import nni.runtime.log
from nni.experiment import Experiment, TrainingServiceConfig
from nni.experiment import management, launcher, rest
from nni.experiment.config import util
from nni.experiment.config.base import ConfigBase, PathLike
from nni.experiment.pipe import Pipe
from nni.tools.nnictl.command_utils import kill_command

from ..codegen import model_to_pytorch_script
from ..converter import convert_to_graph
from ..execution import list_models, set_execution_engine
from ..execution.python import get_mutation_dict
from ..graph import Model, Evaluator
from ..integration import RetiariiAdvisor
from ..mutator import Mutator
from ..nn.pytorch.mutator import process_inline_mutation, extract_mutation_from_pt_module
from ..strategy import BaseStrategy
from ..oneshot.interface import BaseOneShotTrainer

_logger = logging.getLogger(__name__)


@dataclass(init=False)
class RetiariiExeConfig(ConfigBase):
    experiment_name: Optional[str] = None
    search_space: Any = ''  # TODO: remove
    trial_command: str = '_reserved'
    trial_code_directory: PathLike = '.'
    trial_concurrency: int
    trial_gpu_number: int = 0
    max_experiment_duration: Optional[str] = None
    max_trial_number: Optional[int] = None
    nni_manager_ip: Optional[str] = None
    debug: bool = False
    log_level: Optional[str] = None
    experiment_working_directory: PathLike = '~/nni-experiments'
    # remove configuration of tuner/assessor/advisor
    training_service: TrainingServiceConfig
    execution_engine: str = 'py'

    def __init__(self, training_service_platform: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if training_service_platform is not None:
            assert 'training_service' not in kwargs
            self.training_service = util.training_service_config_factory(platform = training_service_platform)
        self.__dict__['trial_command'] = 'python3 -m nni.retiarii.trial_entry py'

    def __setattr__(self, key, value):
        fixed_attrs = {'search_space': '',
                       'trial_command': '_reserved'}
        if key in fixed_attrs and fixed_attrs[key] != value:
            raise AttributeError(f'{key} is not supposed to be set in Retiarii mode by users!')
        # 'trial_code_directory' is handled differently because the path will be converted to absolute path by us
        if key == 'trial_code_directory' and not (value == Path('.') or os.path.isabs(value)):
            raise AttributeError(f'{key} is not supposed to be set in Retiarii mode by users!')
        if key == 'execution_engine':
            assert value in ['base', 'py', 'cgo'], f'The specified execution engine "{value}" is not supported.'
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
    'trial_code_directory': util.canonical_path,
    'max_experiment_duration': lambda value: f'{util.parse_time(value)}s' if value is not None else None,
    'experiment_working_directory': util.canonical_path
}

_validation_rules = {
    'trial_code_directory': lambda value: (Path(value).is_dir(), f'"{value}" does not exist or is not directory'),
    'trial_concurrency': lambda value: value > 0,
    'trial_gpu_number': lambda value: value >= 0,
    'max_experiment_duration': lambda value: util.parse_time(value) > 0,
    'max_trial_number': lambda value: value > 0,
    'log_level': lambda value: value in ["trace", "debug", "info", "warning", "error", "fatal"],
    'training_service': lambda value: (type(value) is not TrainingServiceConfig, 'cannot be abstract base class')
}

def preprocess_model(base_model, trainer, applied_mutators, full_ir=True):
    # TODO: this logic might need to be refactored into execution engine
    if full_ir:
        try:
            script_module = torch.jit.script(base_model)
        except Exception as e:
            _logger.error('Your base model cannot be parsed by torch.jit.script, please fix the following error:')
            raise e
        base_model_ir = convert_to_graph(script_module, base_model)
        # handle inline mutations
        mutators = process_inline_mutation(base_model_ir)
    else:
        base_model_ir, mutators = extract_mutation_from_pt_module(base_model)
    base_model_ir.evaluator = trainer

    if mutators is not None and applied_mutators:
        raise RuntimeError('Have not supported mixed usage of LayerChoice/InputChoice and mutators, '
                            'do not use mutators when you use LayerChoice/InputChoice')
    if mutators is not None:
        applied_mutators = mutators
    return base_model_ir, applied_mutators

def debug_mutated_model(base_model, trainer, applied_mutators):
    """
    Locally run only one trial without launching an experiment for debug purpose, then exit.
    For example, it can be used to quickly check shape mismatch.

    Specifically, it applies mutators (default to choose the first candidate for the choices)
    to generate a new model, then run this model locally.

    Parameters
    ----------
    base_model : nni.retiarii.nn.pytorch.nn.Module
        the base model
    trainer : nni.retiarii.evaluator
        the training class of the generated models
    applied_mutators : list
        a list of mutators that will be applied on the base model for generating a new model
    """
    base_model_ir, applied_mutators = preprocess_model(base_model, trainer, applied_mutators)
    from ..strategy import _LocalDebugStrategy
    strategy = _LocalDebugStrategy()
    strategy.run(base_model_ir, applied_mutators)
    _logger.info('local debug completed!')


class RetiariiExperiment(Experiment):
    def __init__(self, base_model: nn.Module, trainer: Union[Evaluator, BaseOneShotTrainer],
                 applied_mutators: List[Mutator] = None, strategy: BaseStrategy = None):
        # TODO: The current design of init interface of Retiarii experiment needs to be reviewed.
        self.config: RetiariiExeConfig = None
        self.port: Optional[int] = None

        self.base_model = base_model
        self.trainer = trainer
        self.applied_mutators = applied_mutators
        self.strategy = strategy

        self._dispatcher = RetiariiAdvisor()
        self._dispatcher_thread: Optional[Thread] = None
        self._proc: Optional[Popen] = None
        self._pipe: Optional[Pipe] = None

    def _start_strategy(self):
        base_model_ir, self.applied_mutators = preprocess_model(
            self.base_model, self.trainer, self.applied_mutators, full_ir=self.config.execution_engine != 'py')

        _logger.info('Start strategy...')
        self.strategy.run(base_model_ir, self.applied_mutators)
        _logger.info('Strategy exit')
        # TODO: find out a proper way to show no more trial message on WebUI
        #self._dispatcher.mark_experiment_as_ending()

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

        # we will probably need a execution engine factory to make this clean and elegant
        if self.config.execution_engine == 'base':
            from ..execution.base import BaseExecutionEngine
            engine = BaseExecutionEngine()
        elif self.config.execution_engine == 'cgo':
            from ..execution.cgo_engine import CGOExecutionEngine
            engine = CGOExecutionEngine()
        elif self.config.execution_engine == 'py':
            from ..execution.python import PurePythonExecutionEngine
            engine = PurePythonExecutionEngine()
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

    def _create_dispatcher(self):
        return self._dispatcher

    def run(self, config: RetiariiExeConfig = None, port: int = 8080, debug: bool = False) -> str:
        """
        Run the experiment.
        This function will block until experiment finish or error.
        """
        if isinstance(self.trainer, BaseOneShotTrainer):
            self.trainer.fit()
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

    def stop(self) -> None:
        """
        Stop background experiment.
        """
        _logger.info('Stopping experiment, please wait...')
        atexit.unregister(self.stop)
 
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
        if self._dispatcher_thread is not None:
            self._dispatcher.stopping = True
            self._dispatcher_thread.join(timeout=1)

        self.id = None
        self.port = None
        self._proc = None
        self._pipe = None
        self._dispatcher = None
        self._dispatcher_thread = None
        _logger.info('Experiment stopped')

    def export_top_models(self, top_k: int = 1, optimize_mode: str = 'maximize', formatter: str = 'dict') -> Any:
        """
        Export several top performing models.

        For one-shot algorithms, only top-1 is supported. For others, ``optimize_mode`` and ``formatter`` are
        available for customization.

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
        if isinstance(self.trainer, BaseOneShotTrainer):
            assert top_k == 1, 'Only support top_k is 1 for now.'
            return self.trainer.export()
        else:
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
