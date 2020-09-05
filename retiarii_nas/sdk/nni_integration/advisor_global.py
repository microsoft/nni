import importlib
import logging
import os
import subprocess
import sys
from threading import Thread

from .. import utils

_logger = logging.getLogger(__name__)

_advisor_instance = None

def init(advisor):
    global _advisor_instance

    if advisor.command == 'tf':
        _advisor_instance = advisor
        strategy = utils.import_(utils.experiment_config('strategy.scheduler'))
        Thread(target=strategy).start()
        return

    # prerun to register infor to a global variable
    os.environ['RETIARII_PRERUN'] = 'retiarii_prerun'
    full_file_path = advisor.command.split(' ')[1]
    file_name = os.path.basename(full_file_path)
    dir_name = os.path.dirname(full_file_path)
    sys.path.append(dir_name)
    module_name = os.path.splitext(file_name)[0]
    # TODO: this import assumes code is executed
    class_module = importlib.import_module(module_name)
    nas_mode = advisor.command.split(' ')[2]
    class_module.main(nas_mode)

    _logger.info('init advisor')
    _advisor_instance = advisor
    exp_config = utils.experiment_config()
    strategy = utils.import_(exp_config.strategy['scheduler'])
    Thread(target=strategy).start()
    _logger.info('strategy thread started')

def get_advisor():
    return _advisor_instance
