# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import os

from nni.tuner import Tuner
from nni.utils import extract_scalar_reward

_logger = logging.getLogger('NaiveTuner')
_logger.info('start')

_pwd = os.path.dirname(__file__)
_result = open(os.path.join(_pwd, 'tuner_result.txt'), 'w')

class NaiveTuner(Tuner):
    def __init__(self, optimize_mode):
        self.cur = 0
        _logger.info('init')

    def generate_parameters(self, parameter_id, **kwargs):
        self.cur += 1
        _logger.info('generate parameters: %s', self.cur)
        return { 'x': self.cur }

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        reward = extract_scalar_reward(value)
        _logger.info('receive trial result: %s, %s, %s', parameter_id, parameters, reward)
        _result.write('%d %d\n' % (parameters['x'], reward))
        _result.flush()

    def update_search_space(self, search_space):
        _logger.info('update_search_space: %s', search_space)
        with open(os.path.join(_pwd, 'tuner_search_space.json'), 'w') as file_:
            json.dump(search_space, file_)

    def _on_exit(self):
        _result.close()

    def _on_error(self):
        _result.write('ERROR\n')
        _result.close()
