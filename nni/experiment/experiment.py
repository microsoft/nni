import atexit
import logging
import socket
from subprocess import Popen
from threading import Thread
import time
from typing import Optional, overload

import colorama
import psutil

import nni.runtime.log
from nni.runtime.msg_dispatcher import MsgDispatcher
from nni.tuner import Tuner

from .config import ExperimentConfig
from . import launcher
from .pipe import Pipe
from . import rest

nni.runtime.log.init_logger_experiment()
_logger = logging.getLogger('nni.experiment')


class Experiment:
    """
    Create and stop an NNI experiment.

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
            A tuner instance.
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
            Supported value: "local", "remote", "openpai".
        """
        ...

    def __init__(self, tuner: Tuner, config=None, training_service=None):
        self.config: ExperimentConfig
        self.port: Optional[int] = None
        self.tuner: Tuner = tuner
        self._proc: Optional[Popen] = None
        self._pipe: Optional[Pipe] = None
        self._dispatcher: Optional[MsgDispatcher] = None
        self._dispatcher_thread: Optional[Thread] = None

        if isinstance(config, str):
            config, training_service = None, config

        if config is None:
            self.config = ExperimentConfig(training_service)
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
        atexit.register(self.stop)

        if debug:
            logging.getLogger('nni').setLevel(logging.DEBUG)

        self._proc, self._pipe = launcher.start_experiment(self.config, port, debug)
        assert self._proc is not None
        assert self._pipe is not None

        self.port = port  # port will be None if start up failed

        # dispatcher must be launched after pipe initialized
        # the logic to launch dispatcher in background should be refactored into dispatcher api
        self._dispatcher = MsgDispatcher(self.tuner, None)
        self._dispatcher_thread = Thread(target=self._dispatcher.run)
        self._dispatcher_thread.start()

        ips = [self.config.nni_manager_ip]
        for interfaces in psutil.net_if_addrs().values():
            for interface in interfaces:
                if interface.family == socket.AF_INET:
                    ips.append(interface.address)
        ips = [f'http://{ip}:{port}' for ip in ips if ip]
        msg = 'Web UI URLs: ' + colorama.Fore.CYAN + ' '.join(ips)
        _logger.info(msg)

        # TODO: register experiment management metadata


    def stop(self) -> None:
        """
        Stop background experiment.
        """
        _logger.info('Stopping experiment...')
        atexit.unregister(self.stop)

        if self._proc is not None:
            self._proc.kill()
        if self._pipe is not None:
            self._pipe.close()
        if self._dispatcher_thread is not None:
            self._dispatcher.stopping = True
            self._dispatcher_thread.join(timeout=1)

        self.port = None
        self._proc = None
        self._pipe = None
        self._dispatcher = None
        self._dispatcher_thread = None


    def run(self, port: int = 8080, debug: bool = False) -> bool:
        """
        Run the experiment.

        This function will block until experiment finish or error.

        Return `True` when experiment done; or return `False` when experiment failed.
        """
        self.start(port, debug)
        try:
            while True:
                time.sleep(10)
                status = self.get_status()
                if status == 'STOPPED':
                    return True
                if status == 'ERROR':
                    return False
        finally:
            self.stop()


    def get_status(self) -> str:
        if self.port is None:
            raise RuntimeError('Experiment is not running')
        resp = rest.get(self.port, '/check-status')
        return resp['status']
