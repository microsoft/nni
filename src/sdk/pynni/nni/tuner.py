# Copyright (c) Microsoft Corporation. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
# OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ==================================================================================================


import logging
import os

import json_tricks

from .common import init_logger
from .protocol import CommandType, send, receive


init_logger('tuner.log')
_logger = logging.getLogger(__name__)


class Tuner:
    # pylint: disable=no-self-use,unused-argument

    def generate_parameters(self, parameter_id):
        """Returns a set of trial (hyper-)parameters, as a serializable object.
        User code must override either this function or 'generate_multiple_parameters()'.
        parameter_id: int
        """
        raise NotImplementedError('Tuner: generate_parameters not implemented')

    def generate_multiple_parameters(self, parameter_id_list):
        """Returns multiple sets of trial (hyper-)parameters, as iterable of serializable objects.
        Call 'generate_parameters()' by 'count' times by default.
        User code must override either this function or 'generate_parameters()'.
        parameter_id_list: list of int
        """
        return [self.generate_parameters(parameter_id) for parameter_id in parameter_id_list]

    def receive_trial_result(self, parameter_id, parameters, reward):
        """Invoked when a trial reports its final result. Must override.
        parameter_id: int
        parameters: object created by 'generate_parameters()'
        reward: object reported by trial
        """
        raise NotImplementedError('Tuner: receive_trial_result not implemented')

    def receive_customized_trial_result(self, parameter_id, parameters, reward):
        """Invoked when a trial added by WebUI reports its final result. Do nothing by default.
        parameter_id: int
        parameters: object created by user
        reward: object reported by trial
        """
        _logger.info('Customized trial job %s ignored by tuner', parameter_id)

    def update_search_space(self, search_space):
        """Update the search space of tuner. Must override.
        search_space: JSON object
        """
        raise NotImplementedError('Tuner: update_search_space not implemented')

    def load_checkpoint(self, path):
        """Load the checkpoint of tuner.
        path: checkpoint directory for tuner
        """
        _logger.info('Load checkpoint ignored by tuner')

    def save_checkpoint(self, path):
        """Save the checkpoint of tuner.
        path: checkpoint directory for tuner
        """
        _logger.info('Save checkpoint ignored by tuner')

    def request_save_checkpoint(self):
        """Request to save the checkpoint of tuner
        """
        self.save_checkpoint(os.getenv('NNI_CHECKPOINT_DIRECTORY'))

    def run(self):
        """Run the tuner.
        This function will never return unless raise.
        """
        mode = os.getenv('NNI_MODE')
        if mode == 'resume':
            self.load_checkpoint(os.getenv('NNI_CHECKPOINT_DIRECTORY'))
        while _handle_request(self):
            pass
        _logger.info('Terminated by NNI manager')


_next_parameter_id = 0
_trial_params = {}
'''key: trial job ID; value: parameters'''
_customized_parameter_ids = set()


def _create_parameter_id():
    global _next_parameter_id  # pylint: disable=global-statement
    _next_parameter_id += 1
    return _next_parameter_id - 1


def _pack_parameter(parameter_id, params, customized=False):
    _trial_params[parameter_id] = params
    ret = {
        'parameter_id': parameter_id,
        'parameter_source': 'customized' if customized else 'algorithm',
        'parameters': params
    }
    return json_tricks.dumps(ret)


def _handle_request(tuner):
    _logger.debug('waiting receive_message')

    command, data = receive()
    if command is None:
        return False

    _logger.debug(command)
    _logger.debug(data)

    if command is CommandType.Terminate:
        return False

    data = json_tricks.loads(data)

    if command is CommandType.RequestTrialJobs:
        # data: number or trial jobs
        ids = [_create_parameter_id() for _ in range(data)]
        params_list = list(tuner.generate_multiple_parameters(ids))
        assert len(ids) == len(params_list)
        for i, _ in enumerate(ids):
            send(CommandType.NewTrialJob, _pack_parameter(ids[i], params_list[i]))

    elif command is CommandType.ReportMetricData:
        # data: { 'type': 'FINAL', 'parameter_id': ..., 'value': ... }
        if data['type'] == 'FINAL':
            id_ = data['parameter_id']
            if id_ in _customized_parameter_ids:
                tuner.receive_customized_trial_result(id_, _trial_params[id_], data['value'])
            else:
                tuner.receive_trial_result(id_, _trial_params[id_], data['value'])

    elif command is CommandType.UpdateSearchSpace:
        # data: search space
        tuner.update_search_space(data)

    elif command is CommandType.AddCustomizedTrialJob:
        # data: parameters
        id_ = _create_parameter_id()
        _customized_parameter_ids.add(id_)
        send(CommandType.NewTrialJob, _pack_parameter(id_, data, customized=True))

    else:
        raise AssertionError('Unsupported command: %s' % command)

    return True
