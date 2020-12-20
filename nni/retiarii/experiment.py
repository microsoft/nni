import logging
import time

from dataclasses import dataclass
from pathlib import Path
from subprocess import Popen
from threading import Thread
from typing import Any, Optional

from ..experiment import Experiment, TrainingServiceConfig, launcher, rest
from ..experiment.config.base import ConfigBase, PathLike
from ..experiment.config import util
from ..experiment.pipe import Pipe
from .graph import Model
from .utils import get_records
from .integration import RetiariiAdvisor
from .converter import convert_to_graph
from .mutator import Mutator, LayerChoiceMutator, InputChoiceMutator
from .trainer.interface import BaseTrainer
from .strategies.strategy import BaseStrategy

_logger = logging.getLogger(__name__)


@dataclass(init=False)
class RetiariiExeConfig(ConfigBase):
    experiment_name: Optional[str] = None
    search_space: Any = ''  # TODO: remove
    trial_command: str = 'python3 -m nni.retiarii.trial_entry'
    trial_code_directory: PathLike = '.'
    trial_concurrency: int
    trial_gpu_number: int = 0
    max_experiment_duration: Optional[str] = None
    max_trial_number: Optional[int] = None
    nni_manager_ip: Optional[str] = None
    debug: bool = False
    log_level: Optional[str] = None
    experiment_working_directory: Optional[PathLike] = None
    # remove configuration of tuner/assessor/advisor
    training_service: TrainingServiceConfig

    def __init__(self, training_service_platform: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if training_service_platform is not None:
            assert 'training_service' not in kwargs
            self.training_service = util.training_service_config_factory(training_service_platform)

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


class RetiariiExperiment(Experiment):
    def __init__(self, base_model: Model, trainer: BaseTrainer,
                 applied_mutators: Mutator, strategy: BaseStrategy):
        self.config: RetiariiExeConfig = None
        self.port: Optional[int] = None

        self.base_model = base_model
        self.trainer = trainer
        self.applied_mutators = applied_mutators
        self.strategy = strategy
        self.recorded_module_args = get_records()

        self._dispatcher = RetiariiAdvisor()
        self._proc: Optional[Popen] = None
        self._pipe: Optional[Pipe] = None

    def _process_inline_mutation(self, base_model):
        """
        the mutators are order independent
        """
        lc_nodes = base_model.get_nodes_by_type('__torch__.nni.retiarii.nn.pytorch.nn.LayerChoice')
        ic_nodes = base_model.get_nodes_by_type('__torch__.nni.retiarii.nn.pytorch.nn.InputChoice')
        if not lc_nodes and not ic_nodes:
            return None
        applied_mutators = []
        for node in lc_nodes:
            mutator = LayerChoiceMutator(node.name, node.operation.parameters['choices'])
            applied_mutators.append(mutator)
        for node in ic_nodes:
            mutator = InputChoiceMutator(node.name, node.operation.parameters['n_chosen'])
            applied_mutators.append(mutator)
        return applied_mutators

    def _start_strategy(self):
        import torch
        try:
            script_module = torch.jit.script(self.base_model)
        except Exception as e:
            _logger.error('Your base model cannot be parsed by torch.jit.script, please fix the following error:')
            raise e
        base_model = convert_to_graph(script_module, self.base_model, self.recorded_module_args)

        assert id(self.trainer) in self.recorded_module_args
        trainer_config = self.recorded_module_args[id(self.trainer)]
        base_model.apply_trainer(trainer_config['modulename'], trainer_config['args'])

        # handle inline mutations
        mutators = self._process_inline_mutation(base_model)
        if mutators is not None and self.applied_mutators:
            raise RuntimeError('Have not supported mixed usage of LayerChoice/InputChoice and mutators, \
                do not use mutators when you use LayerChoice/InputChoice')
        if mutators is not None:
            self.applied_mutators = mutators

        _logger.info('Starting strategy...')
        Thread(target=self.strategy.run, args=(base_model, self.applied_mutators)).start()
        _logger.info('Strategy started!')

    def start(self, config: RetiariiExeConfig, port: int = 8080, debug: bool = False) -> None:
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
        # FIXME:
        if debug:
            logging.getLogger('nni').setLevel(logging.DEBUG)

        self._proc, self._pipe = launcher.start_experiment(config, port, debug)
        assert self._proc is not None
        assert self._pipe is not None

        self.port = port  # port will be None if start up failed

        # dispatcher must be created after pipe initialized
        # the logic to launch dispatcher in background should be refactored into dispatcher api
        Thread(target=self._dispatcher.run).start()

        self._start_strategy()

        # TODO: register experiment management metadata

    def stop(self) -> None:
        """
        Stop background experiment.
        """
        self._proc.kill()
        self._pipe.close()

        self.port = None
        self._proc = None
        self._pipe = None

    def run(self, config: RetiariiExeConfig, port: int = 8080, debug: bool = False) -> str:
        """
        Run the experiment.
        This function will block until experiment finish or error.
        """
        self.config = config
        self.start(config, port, debug)
        try:
            while True:
                time.sleep(10)
                status = self.get_status()
                # TODO: double check the status
                if status in ['ERROR', 'STOPPED', 'NO_MORE_TRIAL']:
                    return status
        finally:
            self.stop()

    def get_status(self) -> str:
        if self.port is None:
            raise RuntimeError('Experiment is not running')
        resp = rest.get(self.port, '/check-status')
        return resp['status']
