# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
NNI's logging system.

For end users you only need to care about :func:`silence_stdout` and :func:`enable_global_logging`.
Following stuffs are for NNI contributors.

The logging system is initialized on importing ``nni``.
By design it should not have side effects on non-NNI modules' logs,
unless the user explicitly invokes :func:`enable_global_logging`.

The logging system is divided into experiment (user code) part, trial part, and dispatcher part.

Experiment Part
===============

There are two kinds of log handlers here, stdout handler and experiment file handler.

Console
-------

NNI prints log messages of ``INFO`` and above levels to stdout, in colorful format.
This can be turned off with :func:`silence_stdout`.

Logs are automatically colored according to their log level.

One can manually alter the text color with escape sequence.
For example ``logger.error('hello ${GREEN}world')`` will print "hello" in red (error's color) and "world" in green.

The escape sequence affects the words from its position to end of line.
It supports all colors in ``colorama.Fore``.

Please use NNI's escape sequences rather than colorama's because the latter will pollute log files.

Log Files
---------

NNI save log messages to ``~/nni-experiments/<ID>/log/experiment.log``.

It is the experiment classes' role to invoke :func:`start_experiment_logging` and :func:`stop_experiment_logging`.

Only the messages logged during an experiment's life span will be saved to log file.
Logs written before `exp.start()` and after `exp.stop()` will not be saved.

If there are multiple experiments running concurrently,
all logs will be saved to all experiments' log files.

Trial Part
==========

WIP

Dispatcher Part
===============

WIP
"""

from __future__ import annotations

__all__ = ['enable_global_logging', 'silence_stdout']

import logging
import sys
from datetime import datetime
from io import TextIOBase
from logging import FileHandler, Formatter, Handler, LogRecord, Logger, StreamHandler
from pathlib import Path
import string
from typing import Optional

import colorama

from .env_vars import dispatcher_env_vars, trial_env_vars

_colorama_initialized: bool = False
_global_logging_enabled: bool = False

_root_logger: Logger = logging.getLogger('nni')
_root_logger.setLevel(logging.INFO)

_handlers: dict[str, Handler] = {}

_colorful: dict[str, str] = colorama.Fore.__dict__
_colorless: dict[str, str] = {key: '' for key in _colorful.keys()}

def start_stdout_logging() -> None:
    """
    Register the stdout handler.

    This function should be invoked on importing nni.

    It is safe to call it multiple times.
    """
    if '_stdout_' in _handlers:
        return

    handler = StreamHandler(sys.stdout)
    handler.setFormatter(_StdoutFormatter())

    _handlers['_stdout_'] = handler
    _root_logger.addHandler(handler)

def silence_stdout() -> None:
    """
    Stop NNI from printing to stdout.

    By default NNI prints log messages of ``INFO`` and higher levels to console.
    Use this function if you want a clean stdout, or if you want to handle logs by yourself.
    """
    handler = _handlers.pop('_stdout_', None)
    if handler is not None:
        _root_logger.removeHandler(handler)

def start_experiment_logging(experiment_id: str, log_file: Path, level: str) -> None:
    """
    Register the log file handler for an experiment's ``experiment.log``.

    This function should be invoked on starting experiment.
    We don't want to create the experiment folder if the user does not launch it.

    If there are multiple experiments running concurrently,
    log messages will be written to all running experiments' log files.

    It is safe to call it multiple times.
    """
    if experiment_id in _handlers:
        return

    handler = FileHandler(log_file, encoding='utf_8')
    handler.setFormatter(_LogFileFormatter())
    handler.setLevel(level.upper())

    _handlers[experiment_id] = handler
    _root_logger.addHandler(handler)

def stop_experiment_logging(experiment_id: str) -> None:
    """
    Unregister an experiment's ``experiment.log`` handler.
    """
    handler = _handlers.pop(experiment_id, None)
    if handler is not None:
        _root_logger.removeHandler(handler)

def enable_global_logging(enable: bool = True) -> None:
    """
    Let NNI to handle all logs. Useful for debugging.

    By default only NNI's logs are printed to stdout and saved to ``nni-experiments`` log files.
    The function will extend these settings to all modules' logs.

    Use ``enable_global_logging(False)`` to reverse it.
    The log level of root logger will not be reversed though.
    """
    global _global_logging_enabled, _root_logger

    if enable == _global_logging_enabled:
        return

    if enable:
        level = logging.getLogger('nni').getEffectiveLevel()
        logging.getLogger().setLevel(level)

    new_root = logging.getLogger() if enable else logging.getLogger('nni')
    for handler in _handlers.values():
        _root_logger.removeHandler(handler)
        new_root.addHandler(handler)

    _root_logger = new_root
    _global_logging_enabled = enable

class _StdoutFormatter(Formatter):
    def __init__(self):
        fmt = '[%(asctime)s] (%(threadName)s:%(name)s) %(message)s'
        datefmt = '%Y-%m-%d %H:%M:%S'
        super().__init__(fmt, datefmt)

    def formatMessage(self, record: LogRecord) -> str:
        global _colorama_initialized

        if '${' in record.message:  # contains colorful text: "hello ${GREEN}world"
            if not _colorama_initialized:
                colorama.init()
                _colorama_initialized = True
            message = string.Template(record.message).safe_substitute(_colorful)
        else:
            message = record.message

        if record.levelno >= logging.ERROR:
            level = colorama.Fore.RED + 'ERROR: '
        elif record.levelno >= logging.WARNING:
            level = colorama.Fore.YELLOW + 'WARNING: '
        elif record.levelno >= logging.INFO:
            level = colorama.Fore.GREEN
        else:
            level = colorama.Fore.BLUE

        content = level + message + colorama.Style.RESET_ALL
        if record.levelno >= logging.INFO:
            return f'[{record.asctime}] {content}'
        elif record.threadName == 'MainThread':
            return f'[{record.asctime}] {record.name} {content}'
        else:
            return f'[{record.asctime}] {record.threadName}:{record.name} {content}'

class _LogFileFormatter(Formatter):
    def __init__(self):
        fmt = '[%(asctime)s] (%(threadName)s:%(name)s) %(message)s'
        datefmt = '%Y-%m-%d %H:%M:%S'
        super().__init__(fmt, datefmt)

    def formatMessage(self, record: LogRecord) -> str:
        if '${' in record.message:
            message = string.Template(record.message).safe_substitute(_colorless)
        else:
            message = record.message

        if record.threadName == 'MainThread':
            return f'[{record.asctime}] {record.levelname} ({record.name}) {message}'
        else:
            return f'[{record.asctime}] {record.levelname} ({record.threadName}:{record.name}) {message}'


## legacy part ##

handlers = {}

log_format = '[%(asctime)s] %(levelname)s (%(name)s/%(threadName)s) %(message)s'
time_format = '%Y-%m-%d %H:%M:%S'
formatter = Formatter(log_format, time_format)


def _init_logger() -> None:
    """
    This function will (and should only) get invoked on the first time of importing nni (no matter which submodule).
    It will try to detect the running environment and setup logger accordingly.

    The detection should work in most cases but for `nnictl` and `nni.experiment`.
    They will be identified as "standalone" mode and must configure the logger by themselves.
    """
    # I think dispatcher and trial do not need colorful stdout.
    # Remove this when we finish refactor.
    #colorama.init()

    if dispatcher_env_vars.SDK_PROCESS == 'dispatcher':
        _init_logger_dispatcher()
        return

    trial_platform = trial_env_vars.NNI_PLATFORM

    if trial_platform == 'unittest':
        return

    if trial_platform and not trial_env_vars.REUSE_MODE:
        _init_logger_trial()
        return

    start_stdout_logging()


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
    _register_handler(FileHandler(log_path), log_level)


def _init_logger_trial() -> None:
    log_path = _prepare_log_dir(trial_env_vars.NNI_OUTPUT_DIR) / 'trial.log'
    log_file = open(log_path, 'a')
    _register_handler(StreamHandler(log_file), logging.INFO)

    if trial_env_vars.NNI_PLATFORM == 'local':
        sys.stdout = _LogFileWrapper(log_file)


def _prepare_log_dir(path: Path | str) -> Path:
    if path is None:
        return Path()
    ret = Path(path)
    ret.mkdir(parents=True, exist_ok=True)
    return ret

def _register_handler(handler: Handler, level: int, tag: str = '_default_') -> None:
    assert tag not in handlers
    handlers[tag] = handler
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(level)

class _LogFileWrapper(TextIOBase):
    # wrap the logger file so that anything written to it will automatically get formatted

    def __init__(self, log_file: TextIOBase):
        self.file: TextIOBase = log_file
        self.line_buffer: Optional[str] = None
        self.line_start_time: datetime = datetime.fromtimestamp(0)

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
