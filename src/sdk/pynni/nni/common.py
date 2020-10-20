# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from datetime import datetime
from io import TextIOBase
import logging
import sys
import time

log_level_map = {
    'fatal': logging.FATAL,
    'error': logging.ERROR,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG
}

_time_format = '%m/%d/%Y, %I:%M:%S %p'

# FIXME
# This hotfix the bug that querying installed tuners with `package_utils` will activate dispatcher logger.
# This behavior depends on underlying implementation of `nnictl` and is likely to break in future.
_logger_initialized = False

class _LoggerFileWrapper(TextIOBase):
    def __init__(self, logger_file):
        self.file = logger_file

    def write(self, s):
        if s != '\n':
            cur_time = datetime.now().strftime(_time_format)
            self.file.write('[{}] PRINT '.format(cur_time) + s + '\n')
            self.file.flush()
        return len(s)

def init_logger(logger_file_path, log_level_name='info'):
    """Initialize root logger.
    This will redirect anything from logging.getLogger() as well as stdout to specified file.
    logger_file_path: path of logger file (path-like object).
    """
    global _logger_initialized
    if _logger_initialized:
        return
    _logger_initialized = True

    log_level = log_level_map.get(log_level_name, logging.INFO)
    logger_file = open(logger_file_path, 'w')
    fmt = '[%(asctime)s] %(levelname)s (%(name)s/%(threadName)s) %(message)s'
    logging.Formatter.converter = time.localtime
    formatter = logging.Formatter(fmt, _time_format)
    handler = logging.StreamHandler(logger_file)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    # these modules are too verbose
    logging.getLogger('matplotlib').setLevel(log_level)

    sys.stdout = _LoggerFileWrapper(logger_file)

def init_standalone_logger():
    """
    Initialize root logger for standalone mode.
    This will set NNI's log level to INFO and print its log to stdout.
    """
    global _logger_initialized
    if _logger_initialized:
        return
    _logger_initialized = True

    fmt = '[%(asctime)s] %(levelname)s (%(name)s) %(message)s'
    formatter = logging.Formatter(fmt, _time_format)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    nni_logger = logging.getLogger('nni')
    nni_logger.addHandler(handler)
    nni_logger.setLevel(logging.INFO)
    nni_logger.propagate = False

    # Following line does not affect NNI loggers, but without this user's logger won't be able to
    # print log even it's level is set to INFO, so we do it for user's convenience.
    # If this causes any issue in future, remove it and use `logging.info` instead of
    # `logging.getLogger('xxx')` in all examples.
    logging.basicConfig()


_multi_thread = False
_multi_phase = False

def enable_multi_thread():
    global _multi_thread
    _multi_thread = True

def multi_thread_enabled():
    return _multi_thread

def enable_multi_phase():
    global _multi_phase
    _multi_phase = True

def multi_phase_enabled():
    return _multi_phase
