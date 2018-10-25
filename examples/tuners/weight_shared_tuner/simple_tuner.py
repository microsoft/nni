from nni.tuner import Tuner

from threading import Condition


class SimpleTuner(Tuner):
    """
    simple tuner, test for
    """

    def __init__(self):
        super(SimpleTuner, self).__init__()
        self.trial_meta = {}
        self.f_id = None  # father

    def generate_parameters(self, parameter_id):
        if self.f_id is None:
            self.f_id = parameter_id
            sig_cond = Condition()
            sig_cond.acquire()
            self.trial_meta[parameter_id] = {
                'prev_id': 0,
                'id': parameter_id,
                'signal': sig_cond,
                'checksum': None,
                'path': '',
            }
            return {'prev_id': 0}
        else:
            sig_cond = self.trial_meta[self.f_id]['signal']
            sig_cond.wait()
            self.trial_meta[parameter_id] = {
                'id': parameter_id,
                'prev_id': self.f_id,
                'prev_path': self.trial_meta[self.f_id]['path']
            }

    def receive_trial_result(self, parameter_id, parameters, reward):
        if parameter_id == self.f_id:
            self.trial_meta[parameter_id]['checksum'] = reward['checksum']
            self.trial_meta[parameter_id]['path'] = reward['path']
            self.trial_meta[parameter_id]['signal'].release()
        else:
            if reward['checksum'] != self.trial_meta[self.f_id]['checksum'] + str(self.f_id):
                raise ValueError("Inconsistency in weight sharing!!!")

    def update_search_space(self, search_space):
        pass
