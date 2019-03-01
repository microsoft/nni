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
import os
import threading
import time
import math
import pdb
import copy
import logging

import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from nni.protocol import CommandType, send
from nni.msg_dispatcher_base import MsgDispatcherBase
from nni.common import init_logger

from config_generator import CG_BOHB

logger = logging.getLogger('BOHB_Advisor')

_next_parameter_id = 0
_KEY = 'STEPS'


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
    params_id = '_'.join([str(brackets_id),
                          str(brackets_curr_decay),
                          increased_id])
    return params_id

class Bracket():
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
		"""check whether this bracket has sent out all the hyperparameter configurations"""
		return self.no_more_trial

	def get_n_r(self):
		"""return the values of n and r for the next round"""
		return math.floor(self.n / self.eta**self.i), self.r * self.eta**self.i

	def increase_i(self):
		"""i means the ith round. Increase i by 1"""
		self.i += 1
		if self.i > self.bracket_id:
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
		# (TODO: Shufan)Update value and TPE
		"""If the trial is finished and the corresponding round (i.e., i) has all its trials finished,
		it will choose the top k trials for the next round (i.e., i+1)
		Parameters
		----------
		i: int
			the ith round
		"""
		global _KEY # pylint: disable=global-statement
		self.num_finished_configs[i] += 1
		logger.debug('bracket id: %d, round: %d %d, finished: %d, all: %d', self.bracket_id, self.i, i, self.num_finished_configs[i], self.num_configs_to_run[i])
		if self.num_finished_configs[i] >= self.num_configs_to_run[i] \
			and self.no_more_trial is False:
			# choose candidate configs from finished configs to run in the next round
			assert self.i == i + 1
			this_round_perf = self.configs_perf[i]
			if self.optimize_mode is OptimizeMode.Maximize:
				sorted_perf = sorted(this_round_perf.items(), key=lambda kv: kv[1][1], reverse=True) # reverse
			else:
				sorted_perf = sorted(this_round_perf.items(), key=lambda kv: kv[1][1])
			logger.debug('bracket %s next round %s, sorted hyper configs: %s', self.bracket_id, self.i, sorted_perf)
			next_n, next_r = self.get_n_r()
			logger.debug('bracket %s next round %s, next_n=%d, next_r=%d', self.bracket_id, self.i, next_n, next_r)
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

	def get_hyperparameter_configurations(self, num, r, config_generator, search_space):
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
			params = config_generator.get_config(r)
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

def extract_scalar_reward(value, scalar_key='default'):
	"""
	Raises
	------
	RuntimeError
		Incorrect final result: the final result should be float/int,
		or a dict which has a key named "default" whose value is float/int.
	"""
	if isinstance(value, float) or isinstance(value, int):
		reward = value
	elif isinstance(value, dict) and scalar_key in value and isinstance(value[scalar_key], (float, int)):
		reward = value[scalar_key]
	else:
		raise RuntimeError('Incorrect final result: the final result for %s should be float/int, or a dict which has a key named "default" whose value is float/int.' % str(self.__class__)) 
	return reward

class BOHB(MsgDispatcherBase):
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
		super()
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

		# dict of SuccessiveHalving
		self.brackets = dict()
		# all the configs waiting for run
		self.generated_hyper_configs = []
		# all the completed configs
		self.completed_hyper_configs = []

		self.s_max = math.floor(math.log(self.max_budget / self.min_budget, self.eta))
		self.curr_s = self.s_max

	def load_checkpoint(self):
		pass

	def save_checkpont(self):
		pass

	def handle_initialize(self, search_space):
		"""
		Parameters
		----------
		search_space: search space
			search space of this experiment
		"""
		self.handle_update_search_space(search_space)
		self.cg = CG_BOHB(configspace=self.search_space,
						min_points_in_model = self.min_points_in_model,
						top_n_percent=self.top_n_percent,
						num_samples = self.num_samples,
						random_fraction=self.random_fraction,
				 		bandwidth_factor=self.bandwidth_factor,
						min_bandwidth=self.min_bandwidth)
		'''send(CommandType.Initialized, '')  NameError: name '_out_file' is not defined'''
		return True

	def handle_request_trial_jobs(self, num_trial_job):
		"""
		Parameters
		----------
		num_trial_job: int
			number of trial jobs that nni manager ask to generate
		"""
		for _ in range(num_trial_job):
			self._request_one_trial_job()

		return True

	def _request_one_trial_job(self):
		# (TODO:Shufan) slove the problem when get 75/81/+10 no more 4 trial
		"""get one trial job, i.e., one hyperparameter configuration.
		
		Returns
		-------
		dict:
			one hyperparameter configuration
			0: 'parameter_id', id of new hyperparameter
			1: 'parameter_source', 'algorithm'
			2: 'parameters', value of new hyperparameter
		"""
		if not self.generated_hyper_configs:
			if self.curr_s < 0:
				# have tried all configurations
				ret = {
					'parameter_id': '-1_0_0',
					'parameter_source': 'algorithm',
					'parameters': ''
				}
				send(CommandType.NoMoreTrialJobs, json_tricks.dumps(ret))
				return True
			logger.debug('create a new SuccessiveHalving iteration, self.curr_s=%d', self.curr_s)
			self.brackets[self.curr_s] = Bracket(self.curr_s, self.s_max, self.eta, self.max_budget, self.optimize_mode)
			next_n, next_r = self.brackets[self.curr_s].get_n_r()
			logger.debug('new SuccessiveHalving iteration, next_n=%d, next_r=%d', next_n, next_r)
			# rewrite with TPE
			generated_hyper_configs = self.brackets[self.curr_s].get_hyperparameter_configurations(next_n, next_r, self.cg, self.search_space)
			self.generated_hyper_configs = generated_hyper_configs.copy()
			self.curr_s -= 1

		assert self.generated_hyper_configs
		params = self.generated_hyper_configs.pop()
		ret = {
			'parameter_id': params[0],
			'parameter_source': 'algorithm',
			'parameters': params[1]
		}
		print (ret)
		"""send(CommandType.NewTrialJob, json_tricks.dumps(ret)) NameError: name 'json_tricks' is not defined"""

		return True

	def handle_update_search_space(self, search_space):
		"""change json format to ConfigSpace format dict<dict> -> configspace

		Parameters
		----------
		search_space: JSON object
			search space of this experiment

		Returns
		-------
		ConfigSpace:
			search space in ConfigSpace format
		"""
		cs = CS.ConfigurationSpace()
		for var in search_space:
			if search_space[var]["_type"] is "choice":
				cs.add_hyperparameter(CSH.CategoricalHyperparameter(var, choices=search_space[var]["_value"]))
			elif search_space[var]["_type"] is "randint":
				cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(var, lower=0, upper=search_space[var]["_value"][0]))
			elif search_space[var]["_type"] is "uniform":
				cs.add_hyperparameter(CSH.UniformFloatHyperparameter(var, lower=search_space[var]["_value"][0], upper=search_space[var]["_value"][1]))
			elif search_space[var]["_type"] is "quniform":
				cs.add_hyperparameter(CSH.UniformFloatHyperparameter(var, lower=search_space[var]["_value"][0], upper=search_space[var]["_value"][1], q=search_space[var]["_value"][2]))
			elif search_space[var]["_type"] is "loguniform":
				cs.add_hyperparameter(CSH.UniformFloatHyperparameter(var, lower=search_space[var]["_value"][0], upper=search_space[var]["_value"][1], log=True))
			elif search_space[var]["_type"] is "qloguniform":
				cs.add_hyperparameter(CSH.UniformFloatHyperparameter(var, lower=search_space[var]["_value"][0], upper=search_space[var]["_value"][1], q=search_space[var]["_value"][2], log=True))
			elif search_space[var]["_type"] is "normal":
				cs.add_hyperparameter(CSH.NormalFloatHyperparameter(var, mu=search_space[var]["_value"][1], sigma=search_space[var]["_value"][2]))
			elif search_space[var]["_type"] is "qnormal":
				cs.add_hyperparameter(CSH.NormalFloatHyperparameter(var, mu=search_space[var]["_value"][1], sigma=search_space[var]["_value"][2], q=search_space[var]["_value"][3]))
			elif search_space[var]["_type"] is "lognormal":
				cs.add_hyperparameter(CSH.NormalFloatHyperparameter(var, mu=search_space[var]["_value"][1], sigma=search_space[var]["_value"][2], log=True))
			elif search_space[var]["_type"] is "qlognormal":
				cs.add_hyperparameter(CSH.NormalFloatHyperparameter(var, mu=search_space[var]["_value"][1], sigma=search_space[var]["_value"][2], q=search_space[var]["_value"][3], log=True))
			else:
				raise ValueError('unrecognized type in search_space, type is %s", search_space[var]["_type"]')

		self.search_space = cs
		return True

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
		bracket_id, i, _ = hyper_params['parameter_id'].split('_')
		hyper_configs = self.brackets[int(bracket_id)].inform_trial_end(int(i))
		if hyper_configs is not None:
			logger.debug('bracket %s next round %s, hyper_configs: %s', bracket_id, i, hyper_configs)
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

	def handle_report_metric_data(self, results):
		"""

		Parameters
		----------
		results: 
			it is an object which has keys 'parameter_id', 'value', 'trial_job_id', 'type', 'sequence'.
		
		Raises
		------
		ValueError
			Results type not supported
		"""
		value = extract_scalar_reward(results['value'])
		bracket_id, i, _ = results['parameter_id'].split('_')
		bracket_id = int(bracket_id)
		# Use results to update BO, loss, 
		self.cg.new_result(value, _KEY)
		if results['type'] == 'FINAL':
			# sys.maxsize indicates this value is from FINAL metric data, because data['sequence'] from FINAL metric
			# and PERIODICAL metric are independent, thus, not comparable.
			self.brackets[bracket_id].set_config_perf(int(i), results['parameter_id'], sys.maxsize, value)
			self.completed_hyper_configs.append(results)
		elif results['type'] == 'PERIODICAL':
			self.brackets[bracket_id].set_config_perf(int(i), results['parameter_id'], results['sequence'], value)
		else:
			raise ValueError('Results type not supported: {}'.format(results['type']))

		return True

	def handle_add_customized_trial(self, data):
		pass