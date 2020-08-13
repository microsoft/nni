# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import logging
import os
import random
import numpy as np
from schema import Schema, Optional

import nni
from nni import ClassArgsValidator
import nni.parameter_expressions
from nni.tuner import Tuner
from nni.utils import OptimizeMode, extract_scalar_reward, split_index, json2parameter, json2space


logger = logging.getLogger('pbt_tuner_AutoML')


def perturbation(hyperparameter_type, value, resample_probablity, uv, ub, lv, lb, random_state):
    """
    Perturbation for hyperparameters

    Parameters
    ----------
    hyperparameter_type : str
        type of hyperparameter
    value : list
        parameters for sampling hyperparameter
    resample_probability : float
        probability for resampling
    uv : float/int
        upper value after perturbation
    ub : float/int
        upper bound
    lv : float/int
        lower value after perturbation
    lb : float/int
        lower bound
    random_state : RandomState
        random state
    """
    if random.random() < resample_probablity:
        if hyperparameter_type == "choice":
            return value.index(nni.parameter_expressions.choice(value, random_state))
        else:
            return getattr(nni.parameter_expressions, hyperparameter_type)(*(value + [random_state]))
    else:
        if random.random() > 0.5:
            return min(uv, ub)
        else:
            return max(lv, lb)


def exploit_and_explore(bot_trial_info, top_trial_info, factor, resample_probability, epoch, search_space):
    """
    Replace checkpoint of bot_trial with top, and perturb hyperparameters

    Parameters
    ----------
    bot_trial_info : TrialInfo
        bottom model whose parameters should be replaced
    top_trial_info : TrialInfo
        better model
    factor : float
        factor for perturbation
    resample_probability : float
        probability for resampling
    epoch : int
        step of PBTTuner
    search_space : dict
        search_space to keep perturbed hyperparameters in range
    """
    bot_checkpoint_dir = bot_trial_info.checkpoint_dir
    top_hyper_parameters = top_trial_info.hyper_parameters
    hyper_parameters = copy.deepcopy(top_hyper_parameters)
    random_state = np.random.RandomState()
    hyper_parameters['load_checkpoint_dir'] = hyper_parameters['save_checkpoint_dir']
    hyper_parameters['save_checkpoint_dir'] = os.path.join(bot_checkpoint_dir, str(epoch))
    for key in hyper_parameters.keys():
        hyper_parameter = hyper_parameters[key]
        if key == 'load_checkpoint_dir' or key == 'save_checkpoint_dir':
            continue
        elif search_space[key]["_type"] == "choice":
            choices = search_space[key]["_value"]
            ub, uv = len(choices) - 1, choices.index(hyper_parameter) + 1
            lb, lv = 0, choices.index(hyper_parameter) - 1
        elif search_space[key]["_type"] == "randint":
            lb, ub = search_space[key]["_value"][:2]
            ub -= 1
            uv = hyper_parameter + 1
            lv = hyper_parameter - 1
        elif search_space[key]["_type"] == "uniform":
            lb, ub = search_space[key]["_value"][:2]
            perturb = (ub - lb) * factor
            uv = hyper_parameter + perturb
            lv = hyper_parameter - perturb
        elif search_space[key]["_type"] == "quniform":
            lb, ub, q = search_space[key]["_value"][:3]
            multi = round(hyper_parameter / q)
            uv = (multi + 1) * q
            lv = (multi - 1) * q
        elif search_space[key]["_type"] == "loguniform":
            lb, ub = search_space[key]["_value"][:2]
            perturb = (np.log(ub) - np.log(lb)) * factor
            uv = np.exp(min(np.log(hyper_parameter) + perturb, np.log(ub)))
            lv = np.exp(max(np.log(hyper_parameter) - perturb, np.log(lb)))
        elif search_space[key]["_type"] == "qloguniform":
            lb, ub, q = search_space[key]["_value"][:3]
            multi = round(hyper_parameter / q)
            uv = (multi + 1) * q
            lv = (multi - 1) * q
        elif search_space[key]["_type"] == "normal":
            sigma = search_space[key]["_value"][1]
            perturb = sigma * factor
            uv = ub = hyper_parameter + perturb
            lv = lb = hyper_parameter - perturb
        elif search_space[key]["_type"] == "qnormal":
            q = search_space[key]["_value"][2]
            uv = ub = hyper_parameter + q
            lv = lb = hyper_parameter - q
        elif search_space[key]["_type"] == "lognormal":
            sigma = search_space[key]["_value"][1]
            perturb = sigma * factor
            uv = ub = np.exp(np.log(hyper_parameter) + perturb)
            lv = lb = np.exp(np.log(hyper_parameter) - perturb)
        elif search_space[key]["_type"] == "qlognormal":
            q = search_space[key]["_value"][2]
            uv = ub = hyper_parameter + q
            lv, lb = hyper_parameter - q, 1E-10
        else:
            logger.warning("Illegal type to perturb: %s", search_space[key]["_type"])
            continue

        if search_space[key]["_type"] == "choice":
            idx = perturbation(search_space[key]["_type"], search_space[key]["_value"],
                               resample_probability, uv, ub, lv, lb, random_state)
            hyper_parameters[key] = choices[idx]
        else:
            hyper_parameters[key] = perturbation(search_space[key]["_type"], search_space[key]["_value"],
                                                 resample_probability, uv, ub, lv, lb, random_state)
    bot_trial_info.hyper_parameters = hyper_parameters
    bot_trial_info.clean_id()


class TrialInfo:
    """
    Information of each trial, refresh for each epoch

    """

    def __init__(self, checkpoint_dir=None, hyper_parameters=None, parameter_id=None, score=None):
        self.checkpoint_dir = checkpoint_dir
        self.hyper_parameters = hyper_parameters
        self.parameter_id = parameter_id
        self.score = score

    def clean_id(self):
        self.parameter_id = None

class PBTClassArgsValidator(ClassArgsValidator):
    def validate_class_args(self, **kwargs):
        Schema({
            'optimize_mode': self.choices('optimize_mode', 'maximize', 'minimize'),
            Optional('all_checkpoint_dir'): str,
            Optional('population_size'): self.range('population_size', int, 0, 99999),
            Optional('factors'): float,
            Optional('fraction'): float,
        }).validate(kwargs)

class PBTTuner(Tuner):
    def __init__(self, optimize_mode="maximize", all_checkpoint_dir=None, population_size=10, factor=0.2,
                 resample_probability=0.25, fraction=0.2):
        """
        Initialization

        Parameters
        ----------
        optimize_mode : str
            maximize or minimize
        all_checkpoint_dir : str
            directory to store training model checkpoint
        population_size : int
            number of trials for each epoch
        factor : float
            factor for perturbation
        resample_probability : float
            probability for resampling
        fraction : float
            fraction for selecting bottom and top trials
        """
        self.optimize_mode = OptimizeMode(optimize_mode)
        if all_checkpoint_dir is None:
            all_checkpoint_dir = os.getenv('NNI_CHECKPOINT_DIRECTORY')
            logger.info("Checkpoint dir is set to %s by default.", all_checkpoint_dir)
        self.all_checkpoint_dir = all_checkpoint_dir
        self.population_size = population_size
        self.factor = factor
        self.resample_probability = resample_probability
        self.fraction = fraction
        # defined in trial code
        #self.perturbation_interval = perturbation_interval

        self.population = None
        self.pos = -1
        self.param_ids = []
        self.running = {}
        self.finished = []
        self.credit = 0
        self.finished_trials = 0
        self.epoch = 0

        self.searchspace_json = None
        self.space = None

        self.send_trial_callback = None

        logger.info('PBT tuner initialization')

    def update_search_space(self, search_space):
        """
        Get search space

        Parameters
        ----------
        search_space : dict
            Search space
        """
        logger.info('Update search space %s', search_space)
        self.searchspace_json = search_space
        self.space = json2space(self.searchspace_json)

        self.random_state = np.random.RandomState()
        self.population = []
        is_rand = dict()

        for item in self.space:
            is_rand[item] = True

        for i in range(self.population_size):
            hyper_parameters = json2parameter(
                self.searchspace_json, is_rand, self.random_state)
            hyper_parameters = split_index(hyper_parameters)
            checkpoint_dir = os.path.join(self.all_checkpoint_dir, str(i))
            hyper_parameters['load_checkpoint_dir'] = os.path.join(checkpoint_dir, str(self.epoch))
            hyper_parameters['save_checkpoint_dir'] = os.path.join(checkpoint_dir, str(self.epoch))
            self.population.append(TrialInfo(checkpoint_dir=checkpoint_dir, hyper_parameters=hyper_parameters))

    def generate_multiple_parameters(self, parameter_id_list, **kwargs):
        """
        Returns multiple sets of trial (hyper-)parameters, as iterable of serializable objects.

        Parameters
        ----------
        parameter_id_list : list of int
            Unique identifiers for each set of requested hyper-parameters.
            These will later be used in :meth:`receive_trial_result`.
        **kwargs
            Used for send_trial_callback.

        Returns
        -------
        list
            A list of newly generated configurations
        """
        result = []
        self.send_trial_callback = kwargs['st_callback']
        for parameter_id in parameter_id_list:
            had_exception = False
            try:
                logger.debug("generating param for %s", parameter_id)
                res = self.generate_parameters(parameter_id, **kwargs)
            except nni.NoMoreTrialError:
                had_exception = True
            if not had_exception:
                result.append(res)
        return result

    def generate_parameters(self, parameter_id, **kwargs):
        """
        Generate parameters, if no trial configration for now, self.credit plus 1 to send the config later

        Parameters
        ----------
        parameter_id : int
            Unique identifier for requested hyper-parameters.
            This will later be used in :meth:`receive_trial_result`.
        **kwargs
            Not used

        Returns
        -------
        dict
            One newly generated configuration

        """
        if self.pos == self.population_size - 1:
            logger.debug('Credit added by one in parameters request')
            self.credit += 1
            self.param_ids.append(parameter_id)
            raise nni.NoMoreTrialError('No more parameters now.')
        self.pos += 1
        trial_info = self.population[self.pos]
        trial_info.parameter_id = parameter_id
        self.running[parameter_id] = trial_info
        logger.info('Generate parameter : %s', trial_info.hyper_parameters)
        return trial_info.hyper_parameters

    def _proceed_next_epoch(self):
        """
        """
        logger.info('Proceeding to next epoch')
        self.epoch += 1
        self.population = []
        self.pos = -1
        self.running = {}
        #exploit and explore
        reverse = True if self.optimize_mode == OptimizeMode.Maximize else False
        self.finished = sorted(self.finished, key=lambda x: x.score, reverse=reverse)
        cutoff = int(np.ceil(self.fraction * len(self.finished)))
        tops = self.finished[:cutoff]
        bottoms = self.finished[self.finished_trials - cutoff:]
        for bottom in bottoms:
            top = np.random.choice(tops)
            exploit_and_explore(bottom, top, self.factor, self.resample_probability, self.epoch, self.searchspace_json)
        for trial in self.finished:
            if trial not in bottoms:
                trial.clean_id()
                trial.hyper_parameters['load_checkpoint_dir'] = trial.hyper_parameters['save_checkpoint_dir']
                trial.hyper_parameters['save_checkpoint_dir'] = os.path.join(trial.checkpoint_dir, str(self.epoch))
        self.finished_trials = 0
        for _ in range(self.population_size):
            trial_info = self.finished.pop()
            self.population.append(trial_info)
        while self.credit > 0 and self.pos + 1 < len(self.population):
            self.credit -= 1
            self.pos += 1
            parameter_id = self.param_ids.pop()
            trial_info = self.population[self.pos]
            trial_info.parameter_id = parameter_id
            self.running[parameter_id] = trial_info
            self.send_trial_callback(parameter_id, trial_info.hyper_parameters)

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        """
        Receive trial's result. if the number of finished trials equals ``self.population_size``, start the next epoch to
        train the model.

        Parameters
        ----------
        parameter_id : int
            Unique identifier of used hyper-parameters, same with :meth:`generate_parameters`.
        parameters : dict
            Hyper-parameters generated by :meth:`generate_parameters`.
        value : dict
            Result from trial (the return value of :func:`nni.report_final_result`).
        """
        logger.info('Get one trial result, id = %d, value = %s', parameter_id, value)
        value = extract_scalar_reward(value)
        trial_info = self.running.pop(parameter_id, None)
        trial_info.score = value
        self.finished.append(trial_info)
        self.finished_trials += 1
        if self.finished_trials == self.population_size:
            self._proceed_next_epoch()

    def trial_end(self, parameter_id, success, **kwargs):
        """
        Deal with trial failure

        Parameters
        ----------
        parameter_id : int
            Unique identifier for hyper-parameters used by this trial.
        success : bool
            True if the trial successfully completed; False if failed or terminated.
        **kwargs
            Unstable parameters which should be ignored by normal users.
        """
        if success:
            return
        if self.optimize_mode == OptimizeMode.Minimize:
            value = float('inf')
        else:
            value = float('-inf')
        trial_info = self.running.pop(parameter_id, None)
        trial_info.score = value
        self.finished.append(trial_info)
        self.finished_trials += 1
        if self.finished_trials == self.population_size:
            self._proceed_next_epoch()

    def import_data(self, data):
        """
        Parameters
        ----------
        data : json obj
            imported data records

        Returns
        -------
        int
            the start epoch number after data imported, only used for unittest
        """
        if self.running:
            logger.warning("Do not support importing data in the middle of experiment")
            return
        # the following is for experiment resume
        _completed_num = 0
        epoch_data_dict = {}
        for trial_info in data:
            logger.info("Process data record %s / %s", _completed_num, len(data))
            _completed_num += 1
            # simply validate data format
            _params = trial_info["parameter"]
            _value = trial_info['value']
            # assign fake value for failed trials
            if not _value:
                logger.info("Useless trial data, value is %s, skip this trial data.", _value)
                _value = float('inf') if self.optimize_mode == OptimizeMode.Minimize else float('-inf')
            _value = extract_scalar_reward(_value)
            if 'save_checkpoint_dir' not in _params:
                logger.warning("Invalid data record: save_checkpoint_dir is missing, abandon data import.")
                return
            epoch_num = int(os.path.basename(_params['save_checkpoint_dir']))
            if epoch_num not in epoch_data_dict:
                epoch_data_dict[epoch_num] = []
            epoch_data_dict[epoch_num].append((_params, _value))
        if not epoch_data_dict:
            logger.warning("No valid epochs, abandon data import.")
            return
        # figure out start epoch for resume
        max_epoch_num = max(epoch_data_dict, key=int)
        if len(epoch_data_dict[max_epoch_num]) < self.population_size:
            max_epoch_num -= 1
        # If there is no a single complete round, no data to import, start from scratch
        if max_epoch_num < 0:
            logger.warning("No completed epoch, abandon data import.")
            return
        assert len(epoch_data_dict[max_epoch_num]) == self.population_size
        # check existence of trial save checkpoint dir
        for params, _ in epoch_data_dict[max_epoch_num]:
            if not os.path.isdir(params['save_checkpoint_dir']):
                logger.warning("save_checkpoint_dir %s does not exist, data will not be resumed", params['save_checkpoint_dir'])
                return
        # resume data
        self.epoch = max_epoch_num
        self.finished_trials = self.population_size
        for params, value in epoch_data_dict[max_epoch_num]:
            checkpoint_dir = os.path.dirname(params['save_checkpoint_dir'])
            self.finished.append(TrialInfo(checkpoint_dir=checkpoint_dir, hyper_parameters=params, score=value))
        self._proceed_next_epoch()
        logger.info("Successfully import data to PBT tuner, total data: %d, imported data: %d.", len(data), self.population_size)
        logger.info("Start from epoch %d ...", self.epoch)
        return self.epoch # return for test
