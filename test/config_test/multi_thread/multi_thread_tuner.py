# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import time
from nni.tuner import Tuner


class MultiThreadTuner(Tuner):
    def __init__(self):
        self.parent_done = False

    def generate_parameters(self, parameter_id, **kwargs):
        logging.debug('generate_parameters: %s %s', parameter_id, kwargs)
        if parameter_id == 0:
            return {'x': 0}
        else:
            while not self.parent_done:
                logging.debug('parameter_id %s sleeping', parameter_id)
                time.sleep(2)
            logging.debug('parameter_id %s waked up', parameter_id)
            return {'x': 1}

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        logging.debug('receive_trial_result: %s %s %s %s', parameter_id, parameters, value, kwargs)
        if parameter_id == 0:
            self.parent_done = True

    def update_search_space(self, search_space):
        pass
