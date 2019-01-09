import time
from nni.tuner import Tuner


class MultiThreadTuner(Tuner):
    def __init__(self):
        self.parent_done = False

    def generate_parameters(self, parameter_id):
        if parameter_id == 0:
            return {'x': 0}
        else:
            while not self.parent_done:
                time.sleep(2)
            return {'x': 1}

    def receive_trial_result(self, parameter_id, parameters, value):
        if parameter_id == 0:
            self.parent_done = True

    def update_search_space(self, search_space):
        pass
