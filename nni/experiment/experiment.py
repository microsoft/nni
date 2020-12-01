import logging
from subprocess import Popen
import time
from threading import Thread
from typing import Optional, overload, List, Union, Callable

from nni.runtime.msg_dispatcher import MsgDispatcher
from nni.tuner import Tuner

from nni.retiarii.integration import RetiariiAdvisor
from nni.retiarii.converter.graph_gen import convert_to_graph

from .config import ExperimentConfig
from . import launcher
from .pipe import Pipe
from . import rest

_logger = logging.getLogger(__name__)

class Experiment:
    """
    Controls an NNI experiment.

    You may either create a new NNI experiment with construtor and `Experiment.start()`,
    # TODO: or control an existing experiment with `Experiment.connect()`.

    Attributes
    ----------
    config
        Experiment configuration.
    port
        Web UI port of the experiment, or `None` if it is not running.
    """

    @overload
    def __init__(self, tuner: Tuner, config: ExperimentConfig) -> None:
        """
        Prepare an experiment.

        Use `Experiment.start()` to launch it.

        Parameters
        ----------
        tuner
            A tuner instance.  # TODO: accessor / advisor
        config
            Experiment configuration.
        """
        ...

    @overload
    def __init__(self, tuner: Tuner, training_service: str) -> None:
        """
        Prepare an experiment, leaving configuration fields to be set later.

        Example usage::

            experiment = Experiment(my_tuner, 'remote')
            experiment.config.trial_command = 'python3 trial.py'
            experiment.config.machines.append(RemoteMachineConfig(ip=..., user_name=...))
            ...
            experiment.start(8080)

        Parameters
        ----------
        tuner
            A tuner instance.
        training_service
            Name of training service.
            Supported value: "local", "remote", "openpai"/"pai".
        """
        ...

    def __init__(self, tuner: Tuner, config=None, training_service=None):
        self.config: ExperimentConfig
        self.port: Optional[int] = None
        self._dispatcher = MsgDispatcher(tuner, None)
        self._proc: Optional[Popen] = None
        self._pipe: Optional[Pipe] = None

        if isinstance(config, str):
            config, training_service = None, config
        if training_service == 'openpai':
            training_service = 'pai'

        if config is None:
            self.config = ExperimentConfig.create_template(training_service)
        else:
            self.config = config


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
        if debug:
            logging.getLogger('nni').setLevel(logging.DEBUG)

        self._proc, self._pipe = launcher.start_experiment(self.config, port, debug)
        assert self._proc is not None
        assert self._pipe is not None

        self.port = port  # port will be None if start up failed

        # dispatcher must be created after pipe initialized
        # the logic to launch dispatcher in background should be refactored into dispatcher api
        Thread(target=self._dispatcher.run).start()

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


    def run(self, port: int = 8080, debug: bool = False) -> str:
        """
        Run the experiment.

        This function will block until experiment finish or error.
        """
        self.start(port, debug)
        try:
            while True:
                time.sleep(10)
                status = self.get_status()
                if status in ['ERROR', 'STOPPED', 'NO_MORE_TRIAL']:
                    return status
        finally:
            self.stop()


    def get_status(self) -> str:
        if self.port is None:
            raise RuntimeError('Experiment is not running')
        resp = rest.get(self.port, '/check-status')
        return resp['status']


class RetiariiExperiment(Experiment):
    def __init__(self, base_model: 'nn.Module', trainer: 'BaseTrainer',
                 applied_mutators: List['Mutator'], strategy: Union[str, Callable],
                 recorded_module_args = None):
        self.config: ExperimentConfig = None
        self.port: Optional[int] = None

        self.base_model = base_model
        self.trainer = trainer
        self.applied_mutators = applied_mutators
        self.strategy = strategy
        self.recorded_module_args = recorded_module_args # FIXME: remove this argument

        self._dispatcher = RetiariiAdvisor()
        self._proc: Optional[Popen] = None
        self._pipe: Optional[Pipe] = None

        self.strategy = utils.import_(strategy) if isinstance(strategy, str) else strategy

    def _start_strategy(self):
        import torch
        script_module = torch.jit.script(self.base_model)
        base_model = convert_to_graph(script_module, self.base_model, self.recorded_module_args)
        
        _logger.info('Starting strategy...')
        Thread(target=self.strategy, args=(base_model, self.applied_mutators, self.trainer)).start()
        _logger.info('Strategy started!')

    def start(self, config: ExperimentConfig, port: int = 8080, debug: bool = False) -> None:
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

    def run(self, config: ExperimentConfig, port: int = 8080, debug: bool = False) -> str:
        """
        Run the experiment.

        This function will block until experiment finish or error.
        """
        self.start(config, port, debug)
        try:
            while True:
                time.sleep(10)
                status = self.get_status()
                if status in ['ERROR', 'STOPPED', 'NO_MORE_TRIAL']:
                    return status
        finally:
            self.stop()

    def get_status(self) -> str:
        if self.port is None:
            raise RuntimeError('Experiment is not running')
        resp = rest.get(self.port, '/check-status')
        return resp['status']
