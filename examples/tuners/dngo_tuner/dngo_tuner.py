import random
import torch
import numpy as np

from nni.tuner import Tuner
from torch.distributions import Normal
from pybnn import DNGO

def random_archi_generator(nas_ss, random_state):
    '''random
    '''
    chosen_arch = {}
    for key, val in nas_ss.items():
        assert val['_type'] in ['choice', 'uniform'], \
            "Random NAS Tuner only receives NAS search space whose _type is 'choice' or 'uniform'"
        if val['_type'] == 'choice':
            choices = val['_value']
            index = random_state.randint(len(choices))
            chosen_arch[key] = choices[index]
        elif val['_type'] == 'uniform':
            ranges = val['_value']
            num = random.uniform(ranges[0], ranges[1])
            chosen_arch[key] = num
        else:
            raise ValueError('Unknown key %s and value %s' % (key, val))
    return chosen_arch

class DngoTuner(Tuner):

    def __init__(self):

        self.searchspace_json = None
        self.random_state = None
        self.model = DNGO(do_mcmc=False)
        self.first_flag = True
        self.x = []
        self.y = []


    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        '''
        Receive trial's final result.
        parameter_id: int
        parameters: object created by 'generate_parameters()'
        value: final metrics of the trial, including default metric
        '''
        # update DNGO model 
        self.y.append(value)

    def generate_parameters(self, parameter_id, **kwargs):
        '''
        Returns a set of trial (hyper-)parameters, as a serializable object
        parameter_id: int
        '''
        if self.first_flag:
            self.first_flag = False
            first_x = random_archi_generator(self.searchspace_json, self.random_state)
            self.x.append(list(first_x.values()))
            return first_x

        if self.first_flag2:
            self.first_flag2 = False
            first_x2 = random_archi_generator(self.searchspace_json, self.random_state)
            self.x.append(list(first_x2.values()))
            return first_x2
        
        self.model.train(np.array(self.x), np.array(self.y), do_optimize=True)
        # random samples
        candidate_x = []
        for i in range(1000):
            a = random_archi_generator(self.searchspace_json, self.random_state)
            candidate_x.append(a)
        
        x_test = np.array([np.array(list(xi.values())) for xi in candidate_x])
        m, v = self.model.predict(x_test)
        mean = torch.Tensor(m)
        sigma = torch.Tensor(v)
        # u = (mean - torch.Tensor([args.objective]).expand_as(mean)) / sigma
        u = (mean - torch.Tensor([0.95]).expand_as(mean)) / sigma
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        ei = sigma * (updf + u * ucdf)
        
        indices = torch.argsort(ei)
        rev_indices = reversed(indices)
        new_x = candidate_x[0]
        self.x.append(list(new_x.values()))

        return new_x
    

    def update_search_space(self, search_space):
        '''
        Tuners are advised to support updating search space at run-time.
        If a tuner can only set search space once before generating first hyper-parameters,
        it should explicitly document this behaviour.
        search_space: JSON object created by experiment owner
        '''
        # your code implements here.
        self.searchspace_json = search_space
        self.random_state = np.random.RandomState()

        

