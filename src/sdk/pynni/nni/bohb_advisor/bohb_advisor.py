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
bohb_advisor.py
'''

from enum import Enum, unique
import sys
import math
import logging
import json_tricks

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from nni.protocol import CommandType, send
from nni.msg_dispatcher_base import MsgDispatcherBase
from nni.utils import extract_scalar_reward

from .config_generator import CG_BOHB

logger = logging.getLogger('BOHB_Advisor')

_next_parameter_id = 0
_KEY = 'TRIAL_BUDGET'
_epsilon = 1e-6

@unique
class OptimizeMode(Enum):
    """Optimize Mode class"""
    Minimize = 'minimize'
    Maximize = 'maximize'


def create_parameter_id():
    """Create an id

    Returns
    -------
    int
        parameter id
    """
    global _next_parameter_id  # pylint: disable=global-statement
    _next_parameter_id += 1
    return _next_parameter_id - 1


def create_bracket_parameter_id(brackets_id, brackets_curr_decay, increased_id=-1):
    """Create a full id for a specific bracket's hyperparameter configuration

    Parameters
    ----------
    brackets_id: int
        brackets id
    brackets_curr_decay: int
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
    params_id = '_'.join([str(brackets_id),
                          str(brackets_curr_decay),
                          increased_id])
    return params_id


class Bracket():
    """
    A bracket in BOHB, all the information of a bracket is managed by 
    an instance of this class.

    Parameters
    ----------
    s: int
        The current Successive Halving iteration index.
    s_max: int
        total number of Successive Halving iterations
    eta: float
        In each iteration, a complete run of sequential halving is executed. In it,
		after evaluating each configuration on the same subset size, only a fraction of
		1/eta of them 'advances' to the next round.
	max_budget : float
		The largest budget to consider. Needs to be larger than min_budget!
		The budgets will be geometrically distributed
        :math:`a^2 + b^2 = c^2 \sim \eta^k` for :math:`k\in [0, 1, ... , num\_subsets - 1]`.
    optimize_mode: str
        optimize mode, 'maximize' or 'minimize'
    """
    def __init__(self, s, s_max, eta, max_budget, optimize_mode):
        self.s = s
        self.s_max = s_max
        self.eta = eta
        self.max_budget = max_budget
        self.optimize_mode = optimize_mode

        self.n = math.ceil((s_max + 1) * eta**s / (s + 1) - _epsilon)
        self.r = max_budget / eta**s
        self.i = 0
        self.hyper_configs = []         # [ {id: params}, {}, ... ]
        self.configs_perf = []          # [ {id: [seq, acc]}, {}, ... ]
        self.num_configs_to_run = []    # [ n, n, n, ... ]
        self.num_finished_configs = []  # [ n, n, n, ... ]
        self.no_more_trial = False

    def is_completed(self):
        """check whether this bracket has sent out all the hyperparameter configurations"""
        return self.no_more_trial

    def get_n_r(self):
        """return the values of n and r for the next round"""
        return math.floor(self.n / self.eta**self.i + _epsilon), math.floor(self.r * self.eta**self.i +_epsilon)

    def increase_i(self):
        """i means the ith round. Increase i by 1"""
        self.i += 1

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

        Returns
        -------
        new trial or None:
            If we have generated new trials after this trial end, we will return a new trial parameters.
            Otherwise, we will return None.
        """
        global _KEY  # pylint: disable=global-statement
        self.num_finished_configs[i] += 1
        logger.debug('bracket id: %d, round: %d %d, finished: %d, all: %d',
                     self.s, self.i, i, self.num_finished_configs[i], self.num_configs_to_run[i])
        if self.num_finished_configs[i] >= self.num_configs_to_run[i] and self.no_more_trial is False:
            # choose candidate configs from finished configs to run in the next round
            assert self.i == i + 1
            # finish this bracket
            if self.i > self.s:
                self.no_more_trial = True
                return None
            this_round_perf = self.configs_perf[i]
            if self.optimize_mode is OptimizeMode.Maximize:
                sorted_perf = sorted(this_round_perf.items(
                ), key=lambda kv: kv[1][1], reverse=True)  # reverse
            else:
                sorted_perf = sorted(
                    this_round_perf.items(), key=lambda kv: kv[1][1])
            logger.debug(
                'bracket %s next round %s, sorted hyper configs: %s', self.s, self.i, sorted_perf)
            next_n, next_r = self.get_n_r()
            logger.debug('bracket %s next round %s, next_n=%d, next_r=%d',
                         self.s, self.i, next_n, next_r)
            hyper_configs = dict()
            for k in range(next_n):
                params_id = sorted_perf[k][0]
                params = self.hyper_configs[i][params_id]
                params[_KEY] = next_r  # modify r
                # generate new id
                increased_id = params_id.split('_')[-1]
                new_id = create_bracket_parameter_id(
                    self.s, self.i, increased_id)
                hyper_configs[new_id] = params
            self._record_hyper_configs(hyper_configs)
            return [[key, value] for key, value in hyper_configs.items()]
        return None

    def get_hyperparameter_configurations(self, num, r, config_generator):
        """generate num hyperparameter configurations from search space using Bayesian optimization

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
            params_id = create_bracket_parameter_id(self.s, self.i)
            params = config_generator.get_config(r)
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


class BOHB(MsgDispatcherBase):
    """
    BOHB performs robust and efficient hyperparameter optimization
    at scale by combining the speed of Hyperband searches with the
    guidance and guarantees of convergence of Bayesian Optimization.
    Instead of sampling new configurations at random, BOHB uses 
    kernel density estimators to select promising candidates.

    Parameters
    ----------
    optimize_mode: str
        optimize mode, 'maximize' or 'minimize'
    min_budget: float
		The smallest budget to consider. Needs to be positive!
	max_budget: float
		The largest budget to consider. Needs to be larger than min_budget!
		The budgets will be geometrically distributed
        :math:`a^2 + b^2 = c^2 \sim \eta^k` for :math:`k\in [0, 1, ... , num\_subsets - 1]`.
    eta: int
		In each iteration, a complete run of sequential halving is executed. In it,
		after evaluating each configuration on the same subset size, only a fraction of
		1/eta of them 'advances' to the next round.
		Must be greater or equal to 2.
    min_points_in_model: int
		number of observations to start building a KDE. Default 'None' means
		dim+1, the bare minimum.
    top_n_percent: int
		percentage ( between 1 and 99, default 15) of the observations that are considered good.
	num_samples: int
		number of samples to optimize EI (default 64)
	random_fraction: float
		fraction of purely random configurations that are sampled from the
		prior without the model.
	bandwidth_factor: float
		to encourage diversity, the points proposed to optimize EI, are sampled
		from a 'widened' KDE where the bandwidth is multiplied by this factor (default: 3)
	min_bandwidth: float
		to keep diversity, even when all (good) samples have the same value for one of the parameters,
		a minimum bandwidth (Default: 1e-3) is used instead of zero.
    """
    def __init__(self,
                 optimize_mode='maximize',
                 min_budget=1,
                 max_budget=3,
                 eta=3,
                 min_points_in_model=None,
                 top_n_percent=15,
                 num_samples=64,
                 random_fraction=1/3,
                 bandwidth_factor=3,
                 min_bandwidth=1e-3):
        super(BOHB, self).__init__()
        self.optimize_mode = OptimizeMode(optimize_mode)
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.eta = eta
        self.min_points_in_model = min_points_in_model
        self.top_n_percent = top_n_percent
        self.num_samples = num_samples
        self.random_fraction = random_fraction
        self.bandwidth_factor = bandwidth_factor
        self.min_bandwidth = min_bandwidth

        # all the configs waiting for run
        self.generated_hyper_configs = []
        # all the completed configs
        self.completed_hyper_configs = []

        self.s_max = math.floor(
            math.log(self.max_budget / self.min_budget, self.eta) + _epsilon)
        # current bracket(s) number
        self.curr_s = self.s_max
        # In this case, tuner increases self.credit to issue a trial config sometime later.
        self.credit = 0
        self.brackets = dict()
        self.search_space = None
        # [key, value] = [parameter_id, parameter]
        self.parameters = dict()

        # config generator
        self.cg = None

    def load_checkpoint(self):
        pass

    def save_checkpoint(self):
        pass

    def handle_initialize(self, data):
        """Initialize Tuner, including creating Bayesian optimization-based parametric models 
        and search space formations

        Parameters
        ----------
        data: search space
            search space of this experiment

        Raises
        ------
        ValueError
            Error: Search space is None
        """
        logger.info('start to handle_initialize')
        # convert search space jason to ConfigSpace
        self.handle_update_search_space(data)

        # generate BOHB config_generator using Bayesian optimization
        if self.search_space:
            self.cg = CG_BOHB(configspace=self.search_space,
                              min_points_in_model=self.min_points_in_model,
                              top_n_percent=self.top_n_percent,
                              num_samples=self.num_samples,
                              random_fraction=self.random_fraction,
                              bandwidth_factor=self.bandwidth_factor,
                              min_bandwidth=self.min_bandwidth)
        else:
            raise ValueError('Error: Search space is None')
        # generate first brackets
        self.generate_new_bracket()
        send(CommandType.Initialized, '')

    def generate_new_bracket(self):
        """generate a new bracket"""
        logger.debug(
            'start to create a new SuccessiveHalving iteration, self.curr_s=%d', self.curr_s)
        if self.curr_s < 0:
            logger.info("s < 0, Finish this round of Hyperband in BOHB. Generate new round")
            self.curr_s = self.s_max
        self.brackets[self.curr_s] = Bracket(s=self.curr_s, s_max=self.s_max, eta=self.eta,
                                    max_budget=self.max_budget, optimize_mode=self.optimize_mode)
        next_n, next_r = self.brackets[self.curr_s].get_n_r()
        logger.debug(
            'new SuccessiveHalving iteration, next_n=%d, next_r=%d', next_n, next_r)
        # rewrite with TPE
        generated_hyper_configs = self.brackets[self.curr_s].get_hyperparameter_configurations(
            next_n, next_r, self.cg)
        self.generated_hyper_configs = generated_hyper_configs.copy()

    def handle_request_trial_jobs(self, data):
        """recerive the number of request and generate trials

        Parameters
        ----------
        data: int
            number of trial jobs that nni manager ask to generate
        """
        # Receive new request
        self.credit += data

        for _ in range(self.credit):
            self._request_one_trial_job()

    def _request_one_trial_job(self):
        """get one trial job, i.e., one hyperparameter configuration.

        If this function is called, Command will be sent by BOHB:
        a. If there is a parameter need to run, will return "NewTrialJob" with a dict:
        { 
            'parameter_id': id of new hyperparameter
            'parameter_source': 'algorithm'
            'parameters': value of new hyperparameter
        }
        b. If BOHB don't have parameter waiting, will return "NoMoreTrialJobs" with
        {
            'parameter_id': '-1_0_0',
            'parameter_source': 'algorithm',
            'parameters': ''
        }
        """
        if not self.generated_hyper_configs:
            ret = {
                'parameter_id': '-1_0_0',
                'parameter_source': 'algorithm',
                'parameters': ''
            }
            send(CommandType.NoMoreTrialJobs, json_tricks.dumps(ret))
            return
        assert self.generated_hyper_configs
        params = self.generated_hyper_configs.pop()
        ret = {
            'parameter_id': params[0],
            'parameter_source': 'algorithm',
            'parameters': params[1]
        }
        self.parameters[params[0]] = params[1]
        send(CommandType.NewTrialJob, json_tricks.dumps(ret))
        self.credit -= 1

    def handle_update_search_space(self, data):
        """change json format to ConfigSpace format dict<dict> -> configspace

        Parameters
        ----------
        data: JSON object
            search space of this experiment
        """
        search_space = data
        cs = CS.ConfigurationSpace()
        for var in search_space:
            _type = str(search_space[var]["_type"])
            if _type == 'choice':
                cs.add_hyperparameter(CSH.CategoricalHyperparameter(
                    var, choices=search_space[var]["_value"]))
            elif _type == 'randint':
                cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(
                    var, lower=0, upper=search_space[var]["_value"][0]))
            elif _type == 'uniform':
                cs.add_hyperparameter(CSH.UniformFloatHyperparameter(
                    var, lower=search_space[var]["_value"][0], upper=search_space[var]["_value"][1]))
            elif _type == 'quniform':
                cs.add_hyperparameter(CSH.UniformFloatHyperparameter(
                    var, lower=search_space[var]["_value"][0], upper=search_space[var]["_value"][1], 
                    q=search_space[var]["_value"][2]))
            elif _type == 'loguniform':
                cs.add_hyperparameter(CSH.UniformFloatHyperparameter(
                    var, lower=search_space[var]["_value"][0], upper=search_space[var]["_value"][1], 
                    log=True))
            elif _type == 'qloguniform':
                cs.add_hyperparameter(CSH.UniformFloatHyperparameter(
                    var, lower=search_space[var]["_value"][0], upper=search_space[var]["_value"][1], 
                    q=search_space[var]["_value"][2], log=True))
            elif _type == 'normal':
                cs.add_hyperparameter(CSH.NormalFloatHyperparameter(
                    var, mu=search_space[var]["_value"][1], sigma=search_space[var]["_value"][2]))
            elif _type == 'qnormal':
                cs.add_hyperparameter(CSH.NormalFloatHyperparameter(
                    var, mu=search_space[var]["_value"][1], sigma=search_space[var]["_value"][2], 
                    q=search_space[var]["_value"][3]))
            elif _type == 'lognormal':
                cs.add_hyperparameter(CSH.NormalFloatHyperparameter(
                    var, mu=search_space[var]["_value"][1], sigma=search_space[var]["_value"][2], 
                    log=True))
            elif _type == 'qlognormal':
                cs.add_hyperparameter(CSH.NormalFloatHyperparameter(
                    var, mu=search_space[var]["_value"][1], sigma=search_space[var]["_value"][2], 
                    q=search_space[var]["_value"][3], log=True))
            else:
                raise ValueError(
                    'unrecognized type in search_space, type is {}'.format(_type))

        self.search_space = cs

    def handle_trial_end(self, data):
        """receive the information of trial end and generate next configuaration.

        Parameters
        ----------
        data: dict()
            it has three keys: trial_job_id, event, hyper_params
            trial_job_id: the id generated by training service
            event: the job's state
            hyper_params: the hyperparameters (a string) generated and returned by tuner
        """
        logger.debug('Tuner handle trial end, result is %s', data)

        hyper_params = json_tricks.loads(data['hyper_params'])
        s, i, _ = hyper_params['parameter_id'].split('_')
        hyper_configs = self.brackets[int(s)].inform_trial_end(int(i))

        if hyper_configs is not None:
            logger.debug(
                'bracket %s next round %s, hyper_configs: %s', s, i, hyper_configs)
            self.generated_hyper_configs = self.generated_hyper_configs + hyper_configs
            for _ in range(self.credit):
                self._request_one_trial_job()
        # Finish this bracket and generate a new bracket
        elif self.brackets[int(s)].no_more_trial:
            self.curr_s -= 1
            self.generate_new_bracket()
            for _ in range(self.credit):
                self._request_one_trial_job()

    def handle_report_metric_data(self, data):
        """reveice the metric data and update Bayesian optimization with final result

        Parameters
        ----------
        data:
            it is an object which has keys 'parameter_id', 'value', 'trial_job_id', 'type', 'sequence'.

        Raises
        ------
        ValueError
            Data type not supported
        """
        logger.debug('handle report metric data = %s', data)

        assert 'value' in data
        value = extract_scalar_reward(data['value'])
        if self.optimize_mode is OptimizeMode.Maximize:
            reward = -value
        else:
            reward = value
        assert 'parameter_id' in data
        s, i, _ = data['parameter_id'].split('_')

        logger.debug('bracket id = %s, metrics value = %s, type = %s', s, value, data['type'])
        s = int(s)

        assert 'type' in data
        if data['type'] == 'FINAL':
            # and PERIODICAL metric are independent, thus, not comparable.
            assert 'sequence' in data
            self.brackets[s].set_config_perf(
                int(i), data['parameter_id'], sys.maxsize, value)
            self.completed_hyper_configs.append(data)
       
            _parameters = self.parameters[data['parameter_id']]
            _parameters.pop(_KEY)
            # update BO with loss, max_s budget, hyperparameters
            self.cg.new_result(loss=reward, budget=data['sequence'], parameters=_parameters, update_model=True)
        elif data['type'] == 'PERIODICAL':
            self.brackets[s].set_config_perf(
                int(i), data['parameter_id'], data['sequence'], value)
        else:
            raise ValueError(
                'Data type not supported: {}'.format(data['type']))

    def handle_add_customized_trial(self, data):
        pass

    def handle_import_data(self, data):
        """Import additional data for tuning

        Parameters
        ----------
        data:
            a list of dictionarys, each of which has at least two keys, 'parameter' and 'value'

        Raises
        ------
        AssertionError
            data doesn't have required key 'parameter' and 'value'
        """
        _completed_num = 0
        for trial_info in data:
            logger.info("Importing data, current processing progress %s / %s" %(_completed_num, len(data)))
            _completed_num += 1
            assert "parameter" in trial_info
            _params = trial_info["parameter"]
            assert "value" in trial_info
            _value = trial_info['value']
            if not _value:
                logger.info("Useless trial data, value is %s, skip this trial data." %_value)
                continue
            budget_exist_flag = False
            barely_params = dict()
            for keys in _params:
                if keys == _KEY:
                    _budget = _params[keys]
                    budget_exist_flag = True
                else:
                    barely_params[keys] = _params[keys]
            if not budget_exist_flag:
                _budget = self.max_budget
                logger.info("Set \"TRIAL_BUDGET\" value to %s (max budget)" %self.max_budget)
            if self.optimize_mode is OptimizeMode.Maximize:
                reward = -_value
            else:
                reward = _value
            self.cg.new_result(loss=reward, budget=_budget, parameters=barely_params, update_model=True)
        logger.info("Successfully import tuning data to BOHB advisor.")
