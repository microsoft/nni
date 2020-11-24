# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
hyperband_advisor.py
"""

import copy
import logging
import math
import sys

import json_tricks
import numpy as np
from schema import Schema, Optional

from nni import ClassArgsValidator
from nni.runtime.common import multi_phase_enabled
from nni.runtime.msg_dispatcher_base import MsgDispatcherBase
from nni.runtime.protocol import CommandType, send
from nni.utils import NodeType, OptimizeMode, MetricType, extract_scalar_reward
from nni import parameter_expressions

_logger = logging.getLogger(__name__)

_next_parameter_id = 0
_KEY = 'TRIAL_BUDGET'
_epsilon = 1e-6


def create_parameter_id():
    """Create an id

    Returns
    -------
    int
        parameter id
    """
    global _next_parameter_id
    _next_parameter_id += 1
    return _next_parameter_id - 1


def create_bracket_parameter_id(brackets_id, brackets_curr_decay, increased_id=-1):
    """Create a full id for a specific bracket's hyperparameter configuration

    Parameters
    ----------
    brackets_id: string
        brackets id
    brackets_curr_decay:
        brackets curr decay
    increased_id: int
        increased id

    Returns
    -------
    int
        params id
    """
    if increased_id == -1:
        increased_id = str(create_parameter_id())
    params_id = '_'.join([brackets_id,
                          str(brackets_curr_decay),
                          increased_id])
    return params_id


def json2parameter(ss_spec, random_state):
    """Randomly generate values for hyperparameters from hyperparameter space i.e., x.

    Parameters
    ----------
    ss_spec:
        hyperparameter space
    random_state:
        random operator to generate random values

    Returns
    -------
    Parameter:
        Parameters in this experiment
    """
    if isinstance(ss_spec, dict):
        if NodeType.TYPE in ss_spec.keys():
            _type = ss_spec[NodeType.TYPE]
            _value = ss_spec[NodeType.VALUE]
            if _type == 'choice':
                _index = random_state.randint(len(_value))
                chosen_params = json2parameter(ss_spec[NodeType.VALUE][_index], random_state)
            else:
                chosen_params = getattr(parameter_expressions, _type)(*(_value + [random_state]))
        else:
            chosen_params = dict()
            for key in ss_spec.keys():
                chosen_params[key] = json2parameter(ss_spec[key], random_state)
    elif isinstance(ss_spec, list):
        chosen_params = list()
        for _, subspec in enumerate(ss_spec):
            chosen_params.append(json2parameter(subspec, random_state))
    else:
        chosen_params = copy.deepcopy(ss_spec)
    return chosen_params


class Bracket():
    """A bracket in Hyperband, all the information of a bracket is managed by an instance of this class

    Parameters
    ----------
    bracket_id: string
        The id of this bracket, usually be set as '{Hyperband index}-{SH iteration index}'
    s: int
        The current SH iteration index.
    s_max: int
        total number of SH iterations
    eta: float
        In each iteration, a complete run of sequential halving is executed. In it,
		after evaluating each configuration on the same subset size, only a fraction of
		1/eta of them 'advances' to the next round.
    R:
        the budget associated with each stage
    optimize_mode: str
        optimize mode, 'maximize' or 'minimize'
    """

    def __init__(self, bracket_id, s, s_max, eta, R, optimize_mode):
        self.bracket_id = bracket_id
        self.s = s
        self.s_max = s_max
        self.eta = eta
        self.n = math.ceil((s_max + 1) * (eta ** s) / (s + 1) - _epsilon)
        self.r = R / eta ** s
        self.i = 0
        self.hyper_configs = []  # [ {id: params}, {}, ... ]
        self.configs_perf = []  # [ {id: [seq, acc]}, {}, ... ]
        self.num_configs_to_run = []  # [ n, n, n, ... ]
        self.num_finished_configs = []  # [ n, n, n, ... ]
        self.optimize_mode = OptimizeMode(optimize_mode)
        self.no_more_trial = False

    def is_completed(self):
        """check whether this bracket has sent out all the hyperparameter configurations"""
        return self.no_more_trial

    def get_n_r(self):
        """return the values of n and r for the next round"""
        return math.floor(self.n / self.eta ** self.i + _epsilon), math.floor(self.r * self.eta ** self.i + _epsilon)

    def increase_i(self):
        """i means the ith round. Increase i by 1"""
        self.i += 1
        if self.i > self.s:
            self.no_more_trial = True

    def set_config_perf(self, i, parameter_id, seq, value):
        """update trial's latest result with its sequence number, e.g., epoch number or batch number

        Parameters
        ----------
        i: int
            the ith round
        parameter_id: int
            the id of the trial/parameter
        seq: int
            sequence number, e.g., epoch number or batch number
        value: int
            latest result with sequence number seq

        Returns
        -------
        None
        """
        if parameter_id in self.configs_perf[i]:
            if self.configs_perf[i][parameter_id][0] < seq:
                self.configs_perf[i][parameter_id] = [seq, value]
        else:
            self.configs_perf[i][parameter_id] = [seq, value]

    def inform_trial_end(self, i):
        """If the trial is finished and the corresponding round (i.e., i) has all its trials finished,
        it will choose the top k trials for the next round (i.e., i+1)

        Parameters
        ----------
        i: int
            the ith round
        """
        global _KEY
        self.num_finished_configs[i] += 1
        _logger.debug('bracket id: %d, round: %d %d, finished: %d, all: %d', self.bracket_id, self.i, i,
                      self.num_finished_configs[i], self.num_configs_to_run[i])
        if self.num_finished_configs[i] >= self.num_configs_to_run[i] \
                and self.no_more_trial is False:
            # choose candidate configs from finished configs to run in the next round
            assert self.i == i + 1
            this_round_perf = self.configs_perf[i]
            if self.optimize_mode is OptimizeMode.Maximize:
                sorted_perf = sorted(this_round_perf.items(), key=lambda kv: kv[1][1], reverse=True)  # reverse
            else:
                sorted_perf = sorted(this_round_perf.items(), key=lambda kv: kv[1][1])
            _logger.debug('bracket %s next round %s, sorted hyper configs: %s', self.bracket_id, self.i, sorted_perf)
            next_n, next_r = self.get_n_r()
            _logger.debug('bracket %s next round %s, next_n=%d, next_r=%d', self.bracket_id, self.i, next_n, next_r)
            hyper_configs = dict()
            for k in range(next_n):
                params_id = sorted_perf[k][0]
                params = self.hyper_configs[i][params_id]
                params[_KEY] = next_r  # modify r
                # generate new id
                increased_id = params_id.split('_')[-1]
                new_id = create_bracket_parameter_id(self.bracket_id, self.i, increased_id)
                hyper_configs[new_id] = params
            self._record_hyper_configs(hyper_configs)
            return [[key, value] for key, value in hyper_configs.items()]
        return None

    def get_hyperparameter_configurations(self, num, r, searchspace_json, random_state):
        """Randomly generate num hyperparameter configurations from search space

        Parameters
        ----------
        num: int
            the number of hyperparameter configurations

        Returns
        -------
        list
            a list of hyperparameter configurations. Format: [[key1, value1], [key2, value2], ...]
        """
        global _KEY
        assert self.i == 0
        hyperparameter_configs = dict()
        for _ in range(num):
            params_id = create_bracket_parameter_id(self.bracket_id, self.i)
            params = json2parameter(searchspace_json, random_state)
            params[_KEY] = r
            hyperparameter_configs[params_id] = params
        self._record_hyper_configs(hyperparameter_configs)
        return [[key, value] for key, value in hyperparameter_configs.items()]

    def _record_hyper_configs(self, hyper_configs):
        """after generating one round of hyperconfigs, this function records the generated hyperconfigs,
        creates a dict to record the performance when those hyperconifgs are running, set the number of finished configs
        in this round to be 0, and increase the round number.

        Parameters
        ----------
        hyper_configs: list
            the generated hyperconfigs
        """
        self.hyper_configs.append(hyper_configs)
        self.configs_perf.append(dict())
        self.num_finished_configs.append(0)
        self.num_configs_to_run.append(len(hyper_configs))
        self.increase_i()

class HyperbandClassArgsValidator(ClassArgsValidator):
    def validate_class_args(self, **kwargs):
        Schema({
            'optimize_mode': self.choices('optimize_mode', 'maximize', 'minimize'),
            Optional('exec_mode'): self.choices('exec_mode', 'serial', 'parallelism'),
            Optional('R'): int,
            Optional('eta'): int
        }).validate(kwargs)

class Hyperband(MsgDispatcherBase):
    """Hyperband inherit from MsgDispatcherBase rather than Tuner, because it integrates both tuner's functions and assessor's functions.
    This is an implementation that could fully leverage available resources or follow the algorithm process, i.e., high parallelism or serial.
    A single execution of Hyperband takes a finite budget of (s_max + 1)B.

    Parameters
    ----------
    R: int
        the maximum amount of resource that can be allocated to a single configuration
    eta: int
        the variable that controls the proportion of configurations discarded in each round of SuccessiveHalving
    optimize_mode: str
        optimize mode, 'maximize' or 'minimize'
    exec_mode: str
        execution mode, 'serial' or 'parallelism'
    """

    def __init__(self, R=60, eta=3, optimize_mode='maximize', exec_mode='parallelism'):
        """B = (s_max + 1)R"""
        super(Hyperband, self).__init__()
        self.R = R
        self.eta = eta
        self.brackets = dict()  # dict of Bracket
        self.generated_hyper_configs = []  # all the configs waiting for run
        self.completed_hyper_configs = []  # all the completed configs
        self.s_max = math.floor(math.log(self.R, self.eta) + _epsilon)
        self.curr_s = self.s_max
        self.curr_hb = 0
        self.exec_mode = exec_mode
        self.curr_bracket_id = None

        self.searchspace_json = None
        self.random_state = None
        self.optimize_mode = OptimizeMode(optimize_mode)

        # This is for the case that nnimanager requests trial config, but tuner cannot provide immediately.
        # In this case, tuner increases self.credit to issue a trial config sometime later.
        self.credit = 0

        # record the latest parameter_id of the trial job trial_job_id.
        # if there is no running parameter_id, self.job_id_para_id_map[trial_job_id] == None
        # new trial job is added to this dict and finished trial job is removed from it.
        self.job_id_para_id_map = dict()

    def handle_initialize(self, data):
        """callback for initializing the advisor
        Parameters
        ----------
        data: dict
            search space
        """
        self.handle_update_search_space(data)
        send(CommandType.Initialized, '')

    def handle_request_trial_jobs(self, data):
        """
        Parameters
        ----------
        data: int
            number of trial jobs
        """
        self.credit += data

        for _ in range(self.credit):
            self._request_one_trial_job()

    def _request_one_trial_job(self):
        ret = self._get_one_trial_job()
        if ret is not None:
            send(CommandType.NewTrialJob, json_tricks.dumps(ret))
            self.credit -= 1

    def _get_one_trial_job(self):
        """get one trial job, i.e., one hyperparameter configuration."""
        if not self.generated_hyper_configs:
            if self.exec_mode == 'parallelism' or \
               (self.exec_mode == 'serial' and (self.curr_bracket_id is None or self.brackets[self.curr_bracket_id].is_completed())):
                if self.curr_s < 0:
                    self.curr_s = self.s_max
                    self.curr_hb += 1
                _logger.debug('create a new bracket, self.curr_hb=%d, self.curr_s=%d', self.curr_hb, self.curr_s)
                self.curr_bracket_id = '{}-{}'.format(self.curr_hb, self.curr_s)
                self.brackets[self.curr_bracket_id] = Bracket(self.curr_bracket_id, self.curr_s, self.s_max, self.eta, self.R, self.optimize_mode)
                next_n, next_r = self.brackets[self.curr_bracket_id].get_n_r()
                _logger.debug('new bracket, next_n=%d, next_r=%d', next_n, next_r)
                assert self.searchspace_json is not None and self.random_state is not None
                generated_hyper_configs = self.brackets[self.curr_bracket_id].get_hyperparameter_configurations(next_n, next_r,
                                                                                                    self.searchspace_json,
                                                                                                    self.random_state)
                self.generated_hyper_configs = generated_hyper_configs.copy()
                self.curr_s -= 1
            else:
                ret = {
                    'parameter_id': '-1_0_0',
                    'parameter_source': 'algorithm',
                    'parameters': ''
                }
                send(CommandType.NoMoreTrialJobs, json_tricks.dumps(ret))
                return None

        assert self.generated_hyper_configs
        params = self.generated_hyper_configs.pop(0)
        ret = {
            'parameter_id': params[0],
            'parameter_source': 'algorithm',
            'parameters': params[1]
        }
        return ret

    def handle_update_search_space(self, data):
        """data: JSON object, which is search space
        """
        self.searchspace_json = data
        self.random_state = np.random.RandomState()

    def _handle_trial_end(self, parameter_id):
        """
        Parameters
        ----------
        parameter_id: parameter id of the finished config
        """
        bracket_id, i, _ = parameter_id.split('_')
        hyper_configs = self.brackets[bracket_id].inform_trial_end(int(i))
        if hyper_configs is not None:
            _logger.debug('bracket %s next round %s, hyper_configs: %s', bracket_id, i, hyper_configs)
            self.generated_hyper_configs = self.generated_hyper_configs + hyper_configs
        for _ in range(self.credit):
            self._request_one_trial_job()

    def handle_trial_end(self, data):
        """
        Parameters
        ----------
        data: dict()
            it has three keys: trial_job_id, event, hyper_params
            trial_job_id: the id generated by training service
            event: the job's state
            hyper_params: the hyperparameters (a string) generated and returned by tuner
        """
        hyper_params = json_tricks.loads(data['hyper_params'])
        self._handle_trial_end(hyper_params['parameter_id'])
        if data['trial_job_id'] in self.job_id_para_id_map:
            del self.job_id_para_id_map[data['trial_job_id']]

    def handle_report_metric_data(self, data):
        """
        Parameters
        ----------
        data:
            it is an object which has keys 'parameter_id', 'value', 'trial_job_id', 'type', 'sequence'.

        Raises
        ------
        ValueError
            Data type not supported
        """
        if 'value' in data:
            data['value'] = json_tricks.loads(data['value'])
        # multiphase? need to check
        if data['type'] == MetricType.REQUEST_PARAMETER:
            assert multi_phase_enabled()
            assert data['trial_job_id'] is not None
            assert data['parameter_index'] is not None
            assert data['trial_job_id'] in self.job_id_para_id_map
            self._handle_trial_end(self.job_id_para_id_map[data['trial_job_id']])
            ret = self._get_one_trial_job()
            if data['trial_job_id'] is not None:
                ret['trial_job_id'] = data['trial_job_id']
            if data['parameter_index'] is not None:
                ret['parameter_index'] = data['parameter_index']
            self.job_id_para_id_map[data['trial_job_id']] = ret['parameter_id']
            send(CommandType.SendTrialJobParameter, json_tricks.dumps(ret))
        else:
            value = extract_scalar_reward(data['value'])
            bracket_id, i, _ = data['parameter_id'].split('_')

            # add <trial_job_id, parameter_id> to self.job_id_para_id_map here,
            # because when the first parameter_id is created, trial_job_id is not known yet.
            if data['trial_job_id'] in self.job_id_para_id_map:
                assert self.job_id_para_id_map[data['trial_job_id']] == data['parameter_id']
            else:
                self.job_id_para_id_map[data['trial_job_id']] = data['parameter_id']

            if data['type'] == MetricType.FINAL:
                # sys.maxsize indicates this value is from FINAL metric data, because data['sequence'] from FINAL metric
                # and PERIODICAL metric are independent, thus, not comparable.
                self.brackets[bracket_id].set_config_perf(int(i), data['parameter_id'], sys.maxsize, value)
                self.completed_hyper_configs.append(data)
            elif data['type'] == MetricType.PERIODICAL:
                self.brackets[bracket_id].set_config_perf(int(i), data['parameter_id'], data['sequence'], value)
            else:
                raise ValueError('Data type not supported: {}'.format(data['type']))

    def handle_add_customized_trial(self, data):
        pass

    def handle_import_data(self, data):
        pass
