# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge,
# to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
hyperband_tuner.py
'''

from enum import Enum, unique
import math
import copy
import logging
import numpy as np
import json_tricks

from nni.protocol import CommandType, send
from nni.msg_dispatcher_base import MsgDispatcherBase
from nni.common import init_logger
from .. import parameter_expressions

_logger = logging.getLogger(__name__)

_next_parameter_id = 0
_KEY = 'STEPS'

@unique
class OptimizeMode(Enum):
    '''
    Oprimize Mode class
    '''
    Minimize = 'minimize'
    Maximize = 'maximize'

def create_parameter_id():
    '''
    Create an id
    '''
    global _next_parameter_id  # pylint: disable=global-statement
    _next_parameter_id += 1
    return _next_parameter_id - 1

def create_bracket_parameter_id(brackets_id, brackets_curr_decay, increased_id=-1):
    '''
    Create a full id for a specific bracket's hyperparameter configuration
    '''
    if increased_id == -1:
        increased_id = str(create_parameter_id())
    params_id = '_'.join([str(brackets_id),
                          str(brackets_curr_decay),
                          increased_id])
    return params_id

def json2paramater(ss_spec, random_state):
    '''
    Randomly generate values for hyperparameters from hyperparameter space i.e., x.
    ss_spec: hyperparameter space
    random_state: random operator to generate random values
    '''
    if isinstance(ss_spec, dict):
        if '_type' in ss_spec.keys():
            _type = ss_spec['_type']
            _value = ss_spec['_value']
            if _type == 'choice':
                _index = random_state.randint(len(_value))
                chosen_params = json2paramater(ss_spec['_value'][_index], random_state)
            else:
                chosen_params = eval('parameter_expressions.' + # pylint: disable=eval-used
                                     _type)(*(_value + [random_state]))
        else:
            chosen_params = dict()
            for key in ss_spec.keys():
                chosen_params[key] = json2paramater(ss_spec[key], random_state)
    elif isinstance(ss_spec, list):
        chosen_params = list()
        for _, subspec in enumerate(ss_spec):
            chosen_params.append(json2paramater(subspec, random_state))
    else:
        chosen_params = copy.deepcopy(ss_spec)
    return chosen_params

class Bracket():
    '''
    A bracket in Hyperband, all the information of a bracket is managed by an instance of this class
    '''
    def __init__(self, s, s_max, eta, R, optimize_mode):
        self.bracket_id = s
        self.s_max = s_max
        self.eta = eta
        self.n = math.ceil((s_max + 1) * (eta**s) / (s + 1)) # pylint: disable=invalid-name
        self.r = math.ceil(R / eta**s)                       # pylint: disable=invalid-name
        self.i = 0
        self.hyper_configs = []         # [ {id: params}, {}, ... ]
        self.configs_perf = []          # [ {id: [seq, acc]}, {}, ... ]
        self.num_configs_to_run = []    # [ n, n, n, ... ]
        self.num_finished_configs = []  # [ n, n, n, ... ]
        self.optimize_mode = optimize_mode
        self.no_more_trial = False

    def is_completed(self):
        '''
        check whether this bracket has sent out all the hyperparameter configurations
        '''
        return self.no_more_trial

    def get_n_r(self):
        '''
        return the values of n and r for the next round
        '''
        return math.floor(self.n / self.eta**self.i), self.r * self.eta**self.i

    def increase_i(self):
        '''
        i means the ith round. Increase i by 1
        '''
        self.i += 1
        if self.i > self.bracket_id:
            self.no_more_trial = True

    def set_config_perf(self, i, parameter_id, seq, value):
        '''
        update trial's latest result with its sequence number, e.g., epoch number or batch number
        i: the ith round
        parameter_id: the id of the trial/parameter
        seq: sequence number, e.g., epoch number or batch number
        value: latest result with sequence number seq
        '''
        if parameter_id in self.configs_perf[i]:
            # this should always be true if there is no retry in training service
            _logger.debug('assertion: %d %d, %s %s\n',
                          self.configs_perf[i][parameter_id][0],
                          seq,
                          str(type(self.configs_perf[i][parameter_id][0])),
                          str(type(seq)))
            # assert self.configs_perf[i][parameter_id][0] < seq
            if self.configs_perf[i][parameter_id][0] < seq:
                self.configs_perf[i][parameter_id] = [seq, value]
        else:
            self.configs_perf[i][parameter_id] = [seq, value]
            

    def inform_trial_end(self, i):
        '''
        If the trial is finished and the corresponding round (i.e., i) has all its trials finished,
        it will choose the top k trials for the next round (i.e., i+1)
        '''
        global _KEY # pylint: disable=global-statement
        self.num_finished_configs[i] += 1
        _logger.debug('bracket id: %d, round: %d %d, finished: %d, all: %d', self.bracket_id, self.i, i, self.num_finished_configs[i], self.num_configs_to_run[i])
        if self.num_finished_configs[i] >= self.num_configs_to_run[i] \
            and self.no_more_trial is False:
            # choose candidate configs from finished configs to run in the next round
            assert self.i == i + 1
            this_round_perf = self.configs_perf[i]
            if self.optimize_mode is OptimizeMode.Maximize:
                sorted_perf = sorted(this_round_perf.items(), key=lambda kv: kv[1][1], reverse=True) # reverse
            else:
                sorted_perf = sorted(this_round_perf.items(), key=lambda kv: kv[1][1])
            _logger.debug('bracket %s next round %s, sorted hyper configs: %s', self.bracket_id, self.i, sorted_perf)
            next_n, next_r = self.get_n_r()
            _logger.debug('bracket %s next round %s, next_n=%d, next_r=%d', self.bracket_id, self.i, next_n, next_r)
            hyper_configs = dict()
            for k in range(next_n):
                params_id = sorted_perf[k][0]
                params = self.hyper_configs[i][params_id]
                params[_KEY] = next_r # modify r
                # generate new id
                increased_id = params_id.split('_')[-1]
                new_id = create_bracket_parameter_id(self.bracket_id, self.i, increased_id)
                hyper_configs[new_id] = params
            self._record_hyper_configs(hyper_configs)
            return [[key, value] for key, value in hyper_configs.items()]
        return None

    def get_hyperparameter_configurations(self, num, r, searchspace_json, random_state): # pylint: disable=invalid-name
        '''
        Randomly generate num hyperparameter configurations from search space
        num: the number of hyperparameter configurations
        '''
        global _KEY # pylint: disable=global-statement
        assert self.i == 0
        hyperparameter_configs = dict()
        for _ in range(num):
            params_id = create_bracket_parameter_id(self.bracket_id, self.i)
            params = json2paramater(searchspace_json, random_state)
            params[_KEY] = r
            hyperparameter_configs[params_id] = params
        self._record_hyper_configs(hyperparameter_configs)
        return [[key, value] for key, value in hyperparameter_configs.items()]

    def _record_hyper_configs(self, hyper_configs):
        '''
        after generating one round of hyperconfigs, this function records the generated hyperconfigs,
        creates a dict to record the performance when those hyperconifgs are running, set the number of finished configs
        in this round to be 0, and increase the round number.
        '''
        self.hyper_configs.append(hyper_configs)
        self.configs_perf.append(dict())
        self.num_finished_configs.append(0)
        self.num_configs_to_run.append(len(hyper_configs))
        self.increase_i()


class Hyperband(MsgDispatcherBase):
    '''
    Hyperband inherit from MsgDispatcherBase rather than Tuner,
    because it integrates both tuner's functions and assessor's functions.
    This is an implementation that could fully leverage available resources, i.e., high parallelism.
    A single execution of Hyperband takes a finite budget of (s_max + 1)B.
    '''
    def __init__(self, R, eta=3, optimize_mode='maximize'):
        '''
        R: the maximum amount of resource that can be allocated to a single configuration
        eta: the variable that controls the proportion of configurations discarded in each round of SuccessiveHalving
        B = (s_max + 1)R
        '''
        super()
        self.R = R                        # pylint: disable=invalid-name
        self.eta = eta
        self.brackets = dict()            # dict of Bracket
        self.generated_hyper_configs = [] # all the configs waiting for run
        self.completed_hyper_configs = [] # all the completed configs
        self.s_max = math.floor(math.log(self.R, self.eta))
        self.curr_s = self.s_max

        self.searchspace_json = None
        self.random_state = None
        self.optimize_mode = OptimizeMode(optimize_mode)

        # This is for the case that nnimanager requests trial config, but tuner cannot provide immediately.
        # In this case, tuner increases self.credit to issue a trial config sometime later.
        self.credit = 0

    def load_checkpoint(self):
        pass

    def save_checkpont(self):
        pass

    def handle_initialize(self, data):
        '''
        data is search space
        '''
        self.handle_update_search_space(data)
        send(CommandType.Initialized, '')
        return True

    def handle_request_trial_jobs(self, data):
        '''
        data: number of trial jobs
        '''
        for _ in range(data):
            self._request_one_trial_job()

        return True

    def _request_one_trial_job(self):
        '''
        get one trial job, i.e., one hyperparameter configuration.
        '''
        if not self.generated_hyper_configs:
            if self.curr_s < 0:
                # have tried all configurations
                ret = {
                    'parameter_id': '-1_0_0',
                    'parameter_source': 'algorithm',
                    'parameters': ''
                }
                send(CommandType.NoMoreTrialJobs, json_tricks.dumps(ret))
                self.credit += 1
                return True
            _logger.debug('create a new bracket, self.curr_s=%d', self.curr_s)
            self.brackets[self.curr_s] = Bracket(self.curr_s, self.s_max, self.eta, self.R, self.optimize_mode)
            next_n, next_r = self.brackets[self.curr_s].get_n_r()
            _logger.debug('new bracket, next_n=%d, next_r=%d', next_n, next_r)
            assert self.searchspace_json is not None and self.random_state is not None
            generated_hyper_configs = self.brackets[self.curr_s].get_hyperparameter_configurations(next_n, next_r,
                                                                                                   self.searchspace_json,
                                                                                                   self.random_state)
            self.generated_hyper_configs = generated_hyper_configs.copy()
            self.curr_s -= 1

        assert self.generated_hyper_configs
        params = self.generated_hyper_configs.pop()
        ret = {
            'parameter_id': params[0],
            'parameter_source': 'algorithm',
            'parameters': params[1]
        }
        send(CommandType.NewTrialJob, json_tricks.dumps(ret))

        return True

    def handle_update_search_space(self, data):
        '''
        data: JSON object, which is search space
        '''
        self.searchspace_json = data
        self.random_state = np.random.RandomState()

        return True

    def handle_trial_end(self, data):
        '''
        data: it has three keys: trial_job_id, event, hyper_params
            trial_job_id: the id generated by training service
            event: the job's state
            hyper_params: the hyperparameters (a string) generated and returned by tuner
        '''
        hyper_params = json_tricks.loads(data['hyper_params'])
        bracket_id, i, _ = hyper_params['parameter_id'].split('_')
        hyper_configs = self.brackets[int(bracket_id)].inform_trial_end(int(i))
        if hyper_configs is not None:
            _logger.debug('bracket %s next round %s, hyper_configs: %s', bracket_id, i, hyper_configs)
            self.generated_hyper_configs = self.generated_hyper_configs + hyper_configs
            for _ in range(self.credit):
                if not self.generated_hyper_configs:
                    break
                params = self.generated_hyper_configs.pop()
                ret = {
                    'parameter_id': params[0],
                    'parameter_source': 'algorithm',
                    'parameters': params[1]
                }
                send(CommandType.NewTrialJob, json_tricks.dumps(ret))
                self.credit -= 1

        return True

    def handle_report_metric_data(self, data):
        '''
        data: it is an object which has keys 'parameter_id', 'value', 'trial_job_id', 'type', 'sequence'.
        '''
        if data['type'] == 'FINAL':
            self.completed_hyper_configs.append(data)
        elif data['type'] == 'PERIODICAL':
            bracket_id, i, _ = data['parameter_id'].split('_')
            bracket_id = int(bracket_id)
            self.brackets[bracket_id].set_config_perf(int(i), data['parameter_id'], data['sequence'], data['value'])
        else:
            raise ValueError('Data type not supported: {}'.format(data['type']))

        return True

    def handle_add_customized_trial(self, data):
        pass
