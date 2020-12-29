from datetime import datetime
from io import TextIOBase
import logging
from logging import FileHandler, Formatter, Handler, StreamHandler
from pathlib import Path
import sys
import time
from typing import Optional

import colorama

from .env_vars import dispatcher_env_vars, trial_env_vars


def init_logger() -> None:
    """
    This function will (and should only) get invoked on the first time of importing nni (no matter which submodule).
    It will try to detect the running environment and setup logger accordingly.

    The detection should work in most cases but for `nnictl` and `nni.experiment`.
    They will be identified as "standalone" mode and must configure the logger by themselves.
    """
    colorama.init()

    if dispatcher_env_vars.SDK_PROCESS == 'dispatcher':
        _init_logger_dispatcher()
        return

    trial_platform = trial_env_vars.NNI_PLATFORM

    if trial_platform == 'unittest':
        return

    if trial_platform and not trial_env_vars.REUSE_MODE:
        _init_logger_trial()
        return

    _init_logger_standalone()


def init_logger_experiment() -> None:
    """
    Initialize logger for `nni.experiment.Experiment`.

    This function will get invoked after `init_logger()`.
    """
    formatter.format = _colorful_format


time_format = '%Y-%m-%d %H:%M:%S'

formatter = Formatter(
    '[%(asctime)s] %(levelname)s (%(name)s/%(threadName)s) %(message)s',
    time_format
)

def _init_logger_dispatcher() -> None:
    log_level_map = {
        'fatal': logging.CRITICAL,
        'error': logging.ERROR,
        'warning': logging.WARNING,
        'info': logging.INFO,
        'debug': logging.DEBUG,
        'trace': 0
    }

    log_path = _prepare_log_dir(dispatcher_env_vars.NNI_LOG_DIRECTORY) / 'dispatcher.log'
    log_level = log_level_map.get(dispatcher_env_vars.NNI_LOG_LEVEL, logging.INFO)
    _setup_root_logger(FileHandler(log_path), log_level)


def _init_logger_trial() -> None:
    log_path = _prepare_log_dir(trial_env_vars.NNI_OUTPUT_DIR) / 'trial.log'
    log_file = open(log_path, 'w')
    _setup_root_logger(StreamHandler(log_file), logging.INFO)

    if trial_env_vars.NNI_PLATFORM == 'local':
        sys.stdout = _LogFileWrapper(log_file)


def _init_logger_standalone() -> None:
    _setup_nni_logger(StreamHandler(sys.stdout), logging.INFO)

    # Following line does not affect NNI loggers, but without this user's logger won't
    # print log even it's level is set to INFO, so we do it for user's convenience.
    # If this causes any issue in future, remove it and use `logging.info()` instead of
    # `logging.getLogger('xxx').info()` in all examples.
    logging.basicConfig()


def _prepare_log_dir(path: Optional[str]) -> Path:
    if path is None:
        return Path()
    ret = Path(path)
    ret.mkdir(parents=True, exist_ok=True)
    return ret

def _setup_root_logger(handler: Handler, level: int) -> None:
    _setup_logger('', handler, level)

def _setup_nni_logger(handler: Handler, level: int) -> None:
    _setup_logger('nni', handler, level)

def _setup_logger(name: str, handler: Handler, level: int) -> None:
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False

def _colorful_format(record):
    if record.levelno >= logging.ERROR:
        color = colorama.Fore.RED
    elif record.levelno >= logging.WARNING:
        color = colorama.Fore.YELLOW
    elif record.levelno >= logging.INFO:
        color = colorama.Fore.GREEN
    else:
        color = colorama.Fore.BLUE
    msg = color + (record.msg % record.args) + colorama.Style.RESET_ALL
    time = formatter.formatTime(record, time_format)
    if record.levelno < logging.INFO:
        return '[{}] {}:{} {}'.format(time, record.threadName, record.name, msg)
    else:
        return '[{}] {}'.format(time, msg)

class _LogFileWrapper(TextIOBase):
    # wrap the logger file so that anything written to it will automatically get formatted

    def __init__(self, log_file: TextIOBase):
        self.file: TextIOBase = log_file
        self.line_buffer: Optional[str] = None
        self.line_start_time: Optional[datetime] = None

    def write(self, s: str) -> int:
        cur_time = datetime.now()
        if self.line_buffer and (cur_time - self.line_start_time).total_seconds() > 0.1:
            self.flush()

        if self.line_buffer:
            self.line_buffer += s
        else:
            self.line_buffer = s
            self.line_start_time = cur_time

        if '\n' not in s:
            return len(s)

        time_str = cur_time.strftime(time_format)
        lines = self.line_buffer.split('\n')
        for line in lines[:-1]:
            self.file.write(f'[{time_str}] PRINT {line}\n')
        self.file.flush()

        self.line_buffer = lines[-1]
        self.line_start_time = cur_time
        return len(s)

    def flush(self) -> None:
        if self.line_buffer:
            time_str = self.line_start_time.strftime(time_format)
            self.file.write(f'[{time_str}] PRINT {self.line_buffer}\n')
            self.file.flush()
            self.line_buffer = None
