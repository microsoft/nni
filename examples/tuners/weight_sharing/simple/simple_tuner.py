"""
SimpleTuner for Weight Sharing
"""

import logging

from threading import Event
from nni.tuner import Tuner

_logger = logging.getLogger('WeightSharingTuner')


class SimpleTuner(Tuner):
    """
    simple tuner, test for
    """

    def __init__(self):
        super(SimpleTuner, self).__init__()
        self.trial_meta = {}
        self.f_id = None  # father
        self.sig_event = Event()

    def generate_parameters(self, parameter_id):
        if self.f_id is None:
            self.f_id = parameter_id
            self.trial_meta[parameter_id] = {
                'prev_id': 0,
                'id': parameter_id,
                'checksum': None,
                'path': '',
            }
            _logger.info('generate parameter for father trial %s' % parameter_id)
            return {
                'prev_id': 0,
                'id': parameter_id,
            }
        else:
            self.sig_event.wait()
            self.trial_meta[parameter_id] = {
                'id': parameter_id,
                'prev_id': self.f_id,
                'prev_path': self.trial_meta[self.f_id]['path']
            }
            return self.trial_meta[parameter_id]

    def receive_trial_result(self, parameter_id, parameters, reward):
        if parameter_id == self.f_id:
            self.trial_meta[parameter_id]['checksum'] = reward['checksum']
            self.trial_meta[parameter_id]['path'] = reward['path']
            self.sig_event.set()
        else:
            if reward['checksum'] != self.trial_meta[self.f_id]['checksum'] + str(self.f_id):
                raise ValueError("Inconsistency in weight sharing!!!")

    def update_search_space(self, search_space):
        pass
