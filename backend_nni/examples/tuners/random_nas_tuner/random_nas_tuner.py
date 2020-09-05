import numpy as np

from nni.tuner import Tuner


def random_archi_generator(nas_ss, random_state):
    '''random
    '''
    chosen_arch = {}
    for key, val in nas_ss.items():
        assert val['_type'] in ['layer_choice', 'input_choice'], \
            "Random NAS Tuner only receives NAS search space whose _type is 'layer_choice' or 'input_choice'"
        if val['_type'] == 'layer_choice':
            choices = val['_value']
            index = random_state.randint(len(choices))
            chosen_arch[key] = {'_value': choices[index], '_idx': index}
        elif val['_type'] == 'input_choice':
            choices = val['_value']['candidates']
            n_chosen = val['_value']['n_chosen']
            chosen = []
            idxs = []
            for _ in range(n_chosen):
                index = random_state.randint(len(choices))
                chosen.append(choices[index])
                idxs.append(index)
            chosen_arch[key] = {'_value': chosen, '_idx': idxs}
        else:
            raise ValueError('Unknown key %s and value %s' % (key, val))
    return chosen_arch


class RandomNASTuner(Tuner):
    '''RandomNASTuner
    '''

    def __init__(self):
        self.searchspace_json = None
        self.random_state = None

    def update_search_space(self, search_space):
        '''update
        '''
        self.searchspace_json = search_space
        self.random_state = np.random.RandomState()

    def generate_parameters(self, parameter_id, **kwargs):
        '''generate
        '''
        return random_archi_generator(self.searchspace_json, self.random_state)

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        '''receive
        '''
        pass
