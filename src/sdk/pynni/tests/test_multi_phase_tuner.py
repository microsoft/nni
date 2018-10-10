import logging
import random
from io import BytesIO

import nni
import nni.protocol
from nni.protocol import CommandType, send, receive
from nni.multi_phase.multi_phase_tuner import MultiPhaseTuner
from nni.multi_phase.multi_phase_dispatcher import MultiPhaseMsgDispatcher

from unittest import TestCase, main

class NaiveMultiPhaseTuner(MultiPhaseTuner):
    ''' 
    supports only choices
    '''
    def __init__(self):
        self.search_space = None

    def generate_parameters(self, parameter_id, trial_job_id=None):
        """Returns a set of trial (hyper-)parameters, as a serializable object.
        User code must override either this function or 'generate_multiple_parameters()'.
        parameter_id: int
        """
        generated_parameters = {}
        if self.search_space is None:
            raise AssertionError('Search space not specified')
        for k in self.search_space:
            param = self.search_space[k]
            if not param['_type'] == 'choice':
                raise ValueError('Only choice type is supported')
            param_values = param['_value']
            generated_parameters[k] = param_values[random.randint(0, len(param_values)-1)]
        logging.getLogger(__name__).debug(generated_parameters)
        return generated_parameters


    def receive_trial_result(self, parameter_id, parameters, reward, trial_job_id):
        logging.getLogger(__name__).debug('receive_trial_result: {},{},{},{}'.format(parameter_id, parameters, reward, trial_job_id))

    def receive_customized_trial_result(self, parameter_id, parameters, reward, trial_job_id):
        pass

    def update_search_space(self, search_space):
        self.search_space = search_space


_in_buf = BytesIO()
_out_buf = BytesIO()

def _reverse_io():
    _in_buf.seek(0)
    _out_buf.seek(0)
    nni.protocol._out_file = _in_buf
    nni.protocol._in_file = _out_buf

def _restore_io():
    _in_buf.seek(0)
    _out_buf.seek(0)
    nni.protocol._in_file = _in_buf
    nni.protocol._out_file = _out_buf

def _test_tuner():
    _reverse_io()  # now we are sending to Tuner's incoming stream
    send(CommandType.UpdateSearchSpace, "{\"learning_rate\": {\"_value\": [0.0001, 0.001, 0.002, 0.005, 0.01], \"_type\": \"choice\"}, \"optimizer\": {\"_value\": [\"Adam\", \"SGD\"], \"_type\": \"choice\"}}")
    send(CommandType.RequestTrialJobs, '2')
    send(CommandType.ReportMetricData, '{"parameter_id":0,"type":"PERIODICAL","value":10,"trial_job_id":"abc"}')
    send(CommandType.ReportMetricData, '{"parameter_id":1,"type":"FINAL","value":11,"trial_job_id":"abc"}')
    send(CommandType.AddCustomizedTrialJob, '{"param":-1}')
    send(CommandType.ReportMetricData, '{"parameter_id":2,"type":"FINAL","value":22,"trial_job_id":"abc"}')
    send(CommandType.RequestTrialJobs, '1')
    send(CommandType.TrialEnd, '{"trial_job_id":"abc"}')
    _restore_io()

    tuner = NaiveMultiPhaseTuner()
    dispatcher = MultiPhaseMsgDispatcher(tuner)
    dispatcher.run()

    _reverse_io()  # now we are receiving from Tuner's outgoing stream

    command, data = receive()  # this one is customized
    print(command, data)

class MultiPhaseTestCase(TestCase):
    def test_tuner(self):
        _test_tuner()

if __name__ == '__main__':
    main()