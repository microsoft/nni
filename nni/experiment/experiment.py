from typing import overload

from nni.runtime.msg_dispatcher import MsgDispatcher
from nni.tuner import Tuner

from .config import ExperimentConfig
from . import launcher
from .pipe import Pipe

from .config import LocalExperimentConfig

class Experiment:
    """
    Controlls an NNI experiment.

    You may either create a new NNI experiment with construtor and `Experiment.start()`,
    or controll an existing experiment with `Experiment.connect()`.

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
        port
            The port of web UI.
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
        port
            The port of web UI.
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
            self.config = LocalExperimentConfig()
            #self.config = ExperimentConfig._create(training_service)
        else:
            self.config = config


    def start(self, port: int = 8080, debug: bool = False) -> None:
        """
        Start the experiment.

        This method will raise exception on failure.
        If it returns, the experiment should have been successfully started.

        Parameters
        ----------
        port
            The port of web UI.
        debug
            Whether to start in debug mode.
        """
        self.config.validate()

        from threading import Thread

        self._proc, self._pipe = launcher.start_experiment(self.config, port, debug)

        Thread(target = lambda: self._dispatcher.run()).start()

        assert self._proc is not None
        assert self._pipe is not None
        self.port = port  # port will be None if previous line raises
        #management._create(self)


    def stop(self) -> None:
        self._proc.kill()
        self._pipe.close()

        self.port = None
        self._proc = None
        self._pipe = None
