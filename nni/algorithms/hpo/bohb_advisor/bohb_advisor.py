# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

'''
bohb_advisor.py
'''
import sys
import math
import logging
from schema import Schema, Optional
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace.read_and_write import pcs_new

import nni
from nni import ClassArgsValidator
from nni.runtime.tuner_command_channel import CommandType
from nni.runtime.msg_dispatcher_base import MsgDispatcherBase
from nni.utils import OptimizeMode, MetricType, extract_scalar_reward
from nni.runtime.common import multi_phase_enabled

from .config_generator import CG_BOHB

logger = logging.getLogger('BOHB_Advisor')

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


class Bracket:
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
        :math:`a^2 + b^2 = c^2 \\sim \\eta^k` for :math:`k\\in [0, 1, ... , num\\_subsets - 1]`.
    optimize_mode: str
        optimize mode, 'maximize' or 'minimize'
    """
    def __init__(self, s, s_max, eta, max_budget, optimize_mode):
        self.s = s
        self.s_max = s_max
        self.eta = eta
        self.max_budget = max_budget
        self.optimize_mode = OptimizeMode(optimize_mode)

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
        global _KEY
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

class BOHBClassArgsValidator(ClassArgsValidator):
    def validate_class_args(self, **kwargs):
        Schema({
            'optimize_mode': self.choices('optimize_mode', 'maximize', 'minimize'),
            Optional('min_budget'): self.range('min_budget', int, 0, 9999),
            Optional('max_budget'): self.range('max_budget', int, 0, 9999),
            Optional('eta'): self.range('eta', int, 0, 9999),
            Optional('min_points_in_model'): self.range('min_points_in_model', int, 0, 9999),
            Optional('top_n_percent'): self.range('top_n_percent', int, 1, 99),
            Optional('num_samples'): self.range('num_samples', int, 1, 9999),
            Optional('random_fraction'): self.range('random_fraction', float, 0, 9999),
            Optional('bandwidth_factor'): self.range('bandwidth_factor', float, 0, 9999),
            Optional('min_bandwidth'): self.range('min_bandwidth', float, 0, 9999),
            Optional('config_space'): self.path('config_space')
        }).validate(kwargs)

class BOHB(MsgDispatcherBase):
    """
    `BOHB <https://arxiv.org/abs/1807.01774>`__ is a robust and efficient hyperparameter tuning algorithm at scale.
    BO is an abbreviation for "Bayesian Optimization" and HB is an abbreviation for "Hyperband".

    BOHB relies on HB (Hyperband) to determine how many configurations to evaluate with which budget,
    but it replaces the random selection of configurations at the beginning of each HB iteration
    by a model-based search (Bayesian Optimization).
    Once the desired number of configurations for the iteration is reached,
    the standard successive halving procedure is carried out using these configurations.
    It keeps track of the performance of all function evaluations g(x, b) of configurations x
    on all budgets b to use as a basis for our models in later iterations.
    Please refer to the paper :footcite:t:`falkner2018bohb` for detailed algorithm.

    Note that BOHB needs additional installation using the following command:

    .. code-block:: bash

        pip install nni[BOHB]

    Examples
    --------

    .. code-block::

        config.tuner.name = 'BOHB'
        config.tuner.class_args = {
            'optimize_mode': 'maximize',
            'min_budget': 1,
            'max_budget': 27,
            'eta': 3,
            'min_points_in_model': 7,
            'top_n_percent': 15,
            'num_samples': 64,
            'random_fraction': 0.33,
            'bandwidth_factor': 3.0,
            'min_bandwidth': 0.001
        }

    Parameters
    ----------
    optimize_mode: str
        Optimize mode, 'maximize' or 'minimize'.
    min_budget: float
        The smallest budget to assign to a trial job, (budget can be the number of mini-batches or epochs).
        Needs to be positive.
    max_budget: float
        The largest budget to assign to a trial job. Needs to be larger than min_budget.
        The budgets will be geometrically distributed
        :math:`a^2 + b^2 = c^2 \\sim \\eta^k` for :math:`k\\in [0, 1, ... , num\\_subsets - 1]`.
    eta: int
        In each iteration, a complete run of sequential halving is executed. In it,
        after evaluating each configuration on the same subset size, only a fraction of
        1/eta of them 'advances' to the next round.
        Must be greater or equal to 2.
    min_points_in_model: int
        Number of observations to start building a KDE. Default 'None' means dim+1;
        when the number of completed trials in this budget is equal to or larger than ``max{dim+1, min_points_in_model}``,
        BOHB will start to build a KDE model of this budget then use said KDE model to guide configuration selection.
        Needs to be positive. (dim means the number of hyperparameters in search space)
    top_n_percent: int
        Percentage (between 1 and 99, default 15) of the observations which are considered good.
        Good points and bad points are used for building KDE models.
        For example, if you have 100 observed trials and top_n_percent is 15,
        then the top 15% of points will be used for building the good points models "l(x)".
        The remaining 85% of points will be used for building the bad point models "g(x)".
    num_samples: int
        Number of samples to optimize EI (default 64).
        In this case, it will sample "num_samples" points and compare the result of l(x)/g(x).
        Then it will return the one with the maximum l(x)/g(x) value as the next configuration
        if the optimize_mode is ``maximize``. Otherwise, it returns the smallest one.
    random_fraction: float
        Fraction of purely random configurations that are sampled from the prior without the model.
    bandwidth_factor: float
        To encourage diversity, the points proposed to optimize EI are sampled
        from a 'widened' KDE where the bandwidth is multiplied by this factor (default: 3).
        It is suggested to use the default value if you are not familiar with KDE.
    min_bandwidth: float
        To keep diversity, even when all (good) samples have the same value for one of the parameters,
        a minimum bandwidth (default: 1e-3) is used instead of zero.
        It is suggested to use the default value if you are not familiar with KDE.
    config_space: str
        Directly use a .pcs file serialized by `ConfigSpace <https://automl.github.io/ConfigSpace/>` in "pcs new" format.
        In this case, search space file (if provided in config) will be ignored.
        Note that this path needs to be an absolute path. Relative path is currently not supported.

    Notes
    -----

    Below is the introduction of the BOHB process separated in two parts:

    **The first part HB (Hyperband).**
    BOHB follows Hyperband’s way of choosing the budgets and continue to use SuccessiveHalving.
    For more details, you can refer to the :class:`nni.algorithms.hpo.hyperband_advisor.Hyperband`
    and the `reference paper for Hyperband <https://arxiv.org/abs/1603.06560>`__.
    This procedure is summarized by the pseudocode below.

    .. image:: ../../img/bohb_1.png
        :scale: 80 %
        :align: center

    **The second part BO (Bayesian Optimization)**
    The BO part of BOHB closely resembles TPE with one major difference:
    It opted for a single multidimensional KDE compared to the hierarchy of one-dimensional KDEs used in TPE
    in order to better handle interaction effects in the input space.
    Tree Parzen Estimator(TPE): uses a KDE (kernel density estimator) to model the densities.

    .. image:: ../../img/bohb_2.png
        :scale: 80 %
        :align: center

    To fit useful KDEs, we require a minimum number of data points Nmin;
    this is set to d + 1 for our experiments, where d is the number of hyperparameters.
    To build a model as early as possible, we do not wait until Nb = \|Db\|,
    where the number of observations for budget b is large enough to satisfy q · Nb ≥ Nmin.
    Instead, after initializing with Nmin + 2 random configurations, we choose the
    best and worst configurations, respectively, to model the two densities.
    Note that it also samples a constant fraction named **random fraction** of the configurations uniformly at random.

    .. image:: ../../img/bohb_3.png
        :scale: 80 %
        :align: center


    .. image:: ../../img/bohb_6.jpg
        :scale: 65 %
        :align: center

    **The above image shows the workflow of BOHB.**
    Here set max_budget = 9, min_budget = 1, eta = 3, others as default.
    In this case, s_max = 2, so we will continuously run the {s=2, s=1, s=0, s=2, s=1, s=0, ...} cycle.
    In each stage of SuccessiveHalving (the orange box), it will pick the top 1/eta configurations and run them again with more budget,
    repeating the SuccessiveHalving stage until the end of this iteration.
    At the same time, it collects the configurations, budgets and final metrics of each trial
    and use these to build a multidimensional KDEmodel with the key "budget".
    Multidimensional KDE is used to guide the selection of configurations for the next iteration.
    The sampling procedure (using Multidimensional KDE to guide selection) is summarized by the pseudocode below.

    .. image:: ../../img/bohb_4.png
        :scale: 80 %
        :align: center

    **Here is a simple experiment which tunes MNIST with BOHB.**
    Code implementation: :githublink:`examples/trials/mnist-advisor <examples/trials/mnist-advisor>`
    The following is the experimental final results:

    .. image:: ../../img/bohb_5.png
        :scale: 80 %
        :align: center

    More experimental results can be found in the `reference paper <https://arxiv.org/abs/1807.01774>`__.
    It shows that BOHB makes good use of previous results and has a balanced trade-off in exploration and exploitation.
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
                 min_bandwidth=1e-3,
                 config_space=None):
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
        self.config_space = config_space

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

        # record the latest parameter_id of the trial job trial_job_id.
        # if there is no running parameter_id, self.job_id_para_id_map[trial_job_id] == None
        # new trial job is added to this dict and finished trial job is removed from it.
        self.job_id_para_id_map = dict()
        # record the unsatisfied parameter request from trial jobs
        self.unsatisfied_jobs = []

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
        self.send(CommandType.Initialized, '')

    def generate_new_bracket(self):
        """generate a new bracket"""
        logger.debug(
            'start to create a new SuccessiveHalving iteration, self.curr_s=%d', self.curr_s)
        if self.curr_s < 0:
            logger.info("s < 0, Finish this round of Hyperband in BOHB. Generate new round")
            self.curr_s = self.s_max
        self.brackets[self.curr_s] = Bracket(
            s=self.curr_s, s_max=self.s_max, eta=self.eta,
            max_budget=self.max_budget, optimize_mode=self.optimize_mode
        )
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

    def _get_one_trial_job(self):
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
            self.send(CommandType.NoMoreTrialJobs, nni.dump(ret))
            return None
        assert self.generated_hyper_configs
        params = self.generated_hyper_configs.pop(0)
        ret = {
            'parameter_id': params[0],
            'parameter_source': 'algorithm',
            'parameters': params[1]
        }
        self.parameters[params[0]] = params[1]
        return ret

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
        ret = self._get_one_trial_job()
        if ret is not None:
            self.send(CommandType.NewTrialJob, nni.dump(ret))
            self.credit -= 1

    def handle_update_search_space(self, data):
        """change json format to ConfigSpace format dict<dict> -> configspace

        Parameters
        ----------
        data: JSON object
            search space of this experiment
        """
        search_space = data
        cs = None
        logger.debug(f'Received data: {data}')
        if self.config_space:
            logger.info(f'Got a ConfigSpace file path, parsing the search space directly from {self.config_space}. '
                        'The NNI search space is ignored.')
            with open(self.config_space, 'r') as fh:
                cs = pcs_new.read(fh)
        else:
            cs = CS.ConfigurationSpace()
            for var in search_space:
                _type = str(search_space[var]["_type"])
                if _type == 'choice':
                    cs.add_hyperparameter(CSH.CategoricalHyperparameter(
                        var, choices=search_space[var]["_value"]))
                elif _type == 'randint':
                    cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(
                        var, lower=search_space[var]["_value"][0], upper=search_space[var]["_value"][1] - 1))
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
        hyper_params = nni.load(data['hyper_params'])
        if self.is_created_in_previous_exp(hyper_params['parameter_id']):
            # The end of the recovered trial is ignored
            return
        logger.debug('Tuner handle trial end, result is %s', data)
        self._handle_trial_end(hyper_params['parameter_id'])
        if data['trial_job_id'] in self.job_id_para_id_map:
            del self.job_id_para_id_map[data['trial_job_id']]

    def _send_new_trial(self):
        while self.unsatisfied_jobs:
            ret = self._get_one_trial_job()
            if ret is None:
                break
            one_unsatisfied = self.unsatisfied_jobs.pop(0)
            ret['trial_job_id'] = one_unsatisfied['trial_job_id']
            ret['parameter_index'] = one_unsatisfied['parameter_index']
            # update parameter_id in self.job_id_para_id_map
            self.job_id_para_id_map[ret['trial_job_id']] = ret['parameter_id']
            self.send(CommandType.SendTrialJobParameter, nni.dump(ret))
        for _ in range(self.credit):
            self._request_one_trial_job()

    def _handle_trial_end(self, parameter_id):
        s, i, _ = parameter_id.split('_')
        hyper_configs = self.brackets[int(s)].inform_trial_end(int(i))

        if hyper_configs is not None:
            logger.debug(
                'bracket %s next round %s, hyper_configs: %s', s, i, hyper_configs)
            self.generated_hyper_configs = self.generated_hyper_configs + hyper_configs
        # Finish this bracket and generate a new bracket
        elif self.brackets[int(s)].no_more_trial:
            self.curr_s -= 1
            self.generate_new_bracket()
        self._send_new_trial()

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
        if self.is_created_in_previous_exp(data['parameter_id']):
            if data['type'] == MetricType.FINAL:
                # only deal with final metric using import data
                param = self.get_previous_param(data['parameter_id'])
                trial_data = [{'parameter': param, 'value': nni.load(data['value'])}]
                self.handle_import_data(trial_data)
            return
        logger.debug('handle report metric data = %s', data)
        if 'value' in data:
            data['value'] = nni.load(data['value'])
        if data['type'] == MetricType.REQUEST_PARAMETER:
            assert multi_phase_enabled()
            assert data['trial_job_id'] is not None
            assert data['parameter_index'] is not None
            assert data['trial_job_id'] in self.job_id_para_id_map
            self._handle_trial_end(self.job_id_para_id_map[data['trial_job_id']])
            ret = self._get_one_trial_job()
            if ret is None:
                self.unsatisfied_jobs.append({'trial_job_id': data['trial_job_id'], 'parameter_index': data['parameter_index']})
            else:
                ret['trial_job_id'] = data['trial_job_id']
                ret['parameter_index'] = data['parameter_index']
                # update parameter_id in self.job_id_para_id_map
                self.job_id_para_id_map[data['trial_job_id']] = ret['parameter_id']
                self.send(CommandType.SendTrialJobParameter, nni.dump(ret))
        else:
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

            # add <trial_job_id, parameter_id> to self.job_id_para_id_map here,
            # because when the first parameter_id is created, trial_job_id is not known yet.
            if data['trial_job_id'] in self.job_id_para_id_map:
                assert self.job_id_para_id_map[data['trial_job_id']] == data['parameter_id']
            else:
                self.job_id_para_id_map[data['trial_job_id']] = data['parameter_id']

            assert 'type' in data
            if data['type'] == MetricType.FINAL:
                # and PERIODICAL metric are independent, thus, not comparable.
                assert 'sequence' in data
                self.brackets[s].set_config_perf(
                    int(i), data['parameter_id'], sys.maxsize, value)
                self.completed_hyper_configs.append(data)

                _parameters = self.parameters[data['parameter_id']]
                _parameters.pop(_KEY)
                # update BO with loss, max_s budget, hyperparameters
                self.cg.new_result(loss=reward, budget=data['sequence'], parameters=_parameters, update_model=True)
            elif data['type'] == MetricType.PERIODICAL:
                self.brackets[s].set_config_perf(
                    int(i), data['parameter_id'], data['sequence'], value)
            else:
                raise ValueError(
                    'Data type not supported: {}'.format(data['type']))

    def handle_add_customized_trial(self, data):
        global _next_parameter_id
        # data: parameters
        previous_max_param_id = self.recover_parameter_id(data)
        _next_parameter_id = previous_max_param_id + 1

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
        for entry in data:
            entry['value'] = nni.load(entry['value'])
        _completed_num = 0
        for trial_info in data:
            logger.info("Importing data, current processing progress %s / %s", _completed_num, len(data))
            _completed_num += 1
            assert "parameter" in trial_info
            _params = trial_info["parameter"]
            assert "value" in trial_info
            _value = trial_info['value']
            if not _value:
                logger.info("Useless trial data, value is %s, skip this trial data.", _value)
                continue
            _value = extract_scalar_reward(_value)
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
                logger.info("Set \"TRIAL_BUDGET\" value to %s (max budget)", self.max_budget)
            if self.optimize_mode is OptimizeMode.Maximize:
                reward = -_value
            else:
                reward = _value
            self.cg.new_result(loss=reward, budget=_budget, parameters=barely_params, update_model=True)
        logger.info("Successfully import tuning data to BOHB advisor.")
