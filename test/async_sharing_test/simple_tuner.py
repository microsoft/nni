# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
SimpleTuner for Weight Sharing
"""

import logging

from threading import Event, Lock
from nni.tuner import Tuner

_logger = logging.getLogger('WeightSharingTuner')


class SimpleTuner(Tuner):
    """
    simple tuner, test for weight sharing
    """

    def __init__(self):
        super(SimpleTuner, self).__init__()
        self.trial_meta = {}
        self.f_id = None  # father
        self.sig_event = Event()
        self.thread_lock = Lock()

    def generate_parameters(self, parameter_id, **kwargs):
        if self.f_id is None:
            self.thread_lock.acquire()
            self.f_id = parameter_id
            self.trial_meta[parameter_id] = {
                'prev_id': 0,
                'id': parameter_id,
                'checksum': None,
                'path': '',
            }
            _logger.info('generate parameter for father trial %s', parameter_id)
            self.thread_lock.release()
            return {
                'prev_id': 0,
                'id': parameter_id,
            }
        else:
            self.sig_event.wait()
            self.thread_lock.acquire()
            self.trial_meta[parameter_id] = {
                'id': parameter_id,
                'prev_id': self.f_id,
                'prev_path': self.trial_meta[self.f_id]['path']
            }
            self.thread_lock.release()
            return self.trial_meta[parameter_id]

    def receive_trial_result(self, parameter_id, parameters, reward, **kwargs):
        self.thread_lock.acquire()
        if parameter_id == self.f_id:
            self.trial_meta[parameter_id]['checksum'] = reward['checksum']
            self.trial_meta[parameter_id]['path'] = reward['path']
            self.sig_event.set()
        else:
            if reward['checksum'] != self.trial_meta[self.f_id]['checksum']:
                raise ValueError("Inconsistency in weight sharing: {} != {}".format(
                    reward['checksum'], self.trial_meta[self.f_id]['checksum']))
        self.thread_lock.release()

    def update_search_space(self, search_space):
        pass
