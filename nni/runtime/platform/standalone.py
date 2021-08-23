# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import colorama
import logging
import warnings
import json_tricks

__all__ = [
    'get_next_parameter',
    'get_experiment_id',
    'get_trial_id',
    'get_sequence_id',
    'send_metric',
]

_logger = logging.getLogger('nni')


def get_next_parameter():
    warning_message = ''.join([
        colorama.Style.BRIGHT,
        colorama.Fore.RED,
        'Running NNI code without runtime. ',
        'Check the following tutorial if you are new to NNI: ',
        colorama.Fore.YELLOW,
        'https://nni.readthedocs.io/en/stable/Tutorial/QuickStart.html#id1',
        colorama.Style.RESET_ALL
    ])
    warnings.warn(warning_message, RuntimeWarning)
    return {
        'parameter_id': None,
        'parameters': {}
    }

def get_experiment_id():
    return 'STANDALONE'

def get_trial_id():
    return 'STANDALONE'

def get_sequence_id():
    return 0

def send_metric(string):
    metric = json_tricks.loads(string)
    if metric['type'] == 'FINAL':
        _logger.info('Final result: %s', metric['value'])
    elif metric['type'] == 'PERIODICAL':
        _logger.info('Intermediate result: %s  (Index %s)', metric['value'], metric['sequence'])
    else:
        _logger.error('Unexpected metric: %s', string)
