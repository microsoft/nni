# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import json_tricks

from ..common import init_standalone_logger

__all__ = [
    'get_next_parameter',
    'get_experiment_id',
    'get_trial_id',
    'get_sequence_id',
    'send_metric',
]

init_standalone_logger()
_logger = logging.getLogger('nni')


def get_next_parameter():
    _logger.warning('Requesting parameter without NNI framework, returning empty dict')
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
