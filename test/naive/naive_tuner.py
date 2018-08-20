import json
import logging

from nni.tuner import Tuner

_logger = logging.getLogger('NaiveTuner')
_logger.info('start')
_result = open('tuner_result.txt', 'w')

class NaiveTuner(Tuner):
    def __init__(self):
        self.cur = 0
        _logger.info('init')

    def generate_parameters(self, parameter_id):
        self.cur += 1
        _logger.info('generate parameters: %s' % self.cur)
        return { 'x': self.cur }

    def receive_trial_result(self, parameter_id, parameters, reward):
        _logger.info('receive trial result: %s, %s, %s' % (parameter_id, parameters, reward))
        _result.write('%d %d\n' % (parameters['x'], reward))
        _result.flush()

    def update_search_space(self, search_space):
        _logger.info('update_search_space: %s' % search_space)
        with open('tuner_search_space.json', 'w') as file_:
            json.dump(search_space, file_)

try:
    NaiveTuner().run()
    _result.write('DONE\n')
except Exception as e:
    _logger.exception(e)
    _result.write('ERROR\n')
_result.close()
