import logging

from dataclasses import dataclass
from pathlib import Path
from subprocess import Popen
from threading import Thread
from typing import Any, Optional

from ..experiment import Experiment, TrainingServiceConfig
from ..experiment.config.base import ConfigBase, PathLike
from ..experiment.config import util
from ..experiment.pipe import Pipe

from .graph import Model
from .utils import get_records
from .integration import RetiariiAdvisor
from .converter import convert_to_graph
from .mutator import Mutator
from .trainer.interface import BaseTrainer, BaseOneShotTrainer
from .strategies.strategy import BaseStrategy
from .trainer import BaseOneShotTrainer

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
            self.training_service = util.training_service_config_factory(platform = training_service_platform)

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
                 applied_mutators: Mutator = None, strategy: BaseStrategy = None):
        self.config: RetiariiExeConfig = None
        self.port: Optional[int] = None

        self.base_model = base_model
        self.trainer = trainer
        self.applied_mutators = applied_mutators
        self.strategy = strategy
        self.recorded_module_args = get_records()

        self._dispatcher = RetiariiAdvisor()
        self._dispatcher_thread: Optional[Thread] = None
        self._proc: Optional[Popen] = None
        self._pipe: Optional[Pipe] = None

    def _start_strategy(self):
        import torch
        from .nn.pytorch.mutator import process_inline_mutation

        try:
            script_module = torch.jit.script(self.base_model)
        except Exception as e:
            _logger.error('Your base model cannot be parsed by torch.jit.script, please fix the following error:')
            raise e
        base_model_ir = convert_to_graph(script_module, self.base_model)

        recorded_module_args = get_records()
        if id(self.trainer) not in recorded_module_args:
            raise KeyError('Your trainer is not found in registered classes. You might have forgotten to \
                register your customized trainer with @register_trainer decorator.')
        trainer_config = recorded_module_args[id(self.trainer)]
        base_model_ir.apply_trainer(trainer_config['modulename'], trainer_config['args'])

        # handle inline mutations
        mutators = process_inline_mutation(base_model_ir)
        if mutators is not None and self.applied_mutators:
            raise RuntimeError('Have not supported mixed usage of LayerChoice/InputChoice and mutators, \
                do not use mutators when you use LayerChoice/InputChoice')
        if mutators is not None:
            self.applied_mutators = mutators

        _logger.info('Starting strategy...')
        Thread(target=self.strategy.run, args=(base_model_ir, self.applied_mutators)).start()
        _logger.info('Strategy started!')

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
        super().start(port, debug)
        self._start_strategy()

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
            super().run(port, debug)

    def export_top_models(self, top_n: int = 1):
        """
        export several top performing models
        """
        if top_n != 1:
            _logger.warning('Only support top_n is 1 for now.')
        if isinstance(self.trainer, BaseOneShotTrainer):
            return self.trainer.export()
        else:
            _logger.info('For this experiment, you can find out the best one from WebUI.')

    def retrain_model(self, model):
        """
        this function retrains the exported model, and test it to output test accuracy
        """
        raise NotImplementedError
