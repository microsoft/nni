# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import logging
import os
import numpy as np

import nni
from nni.tuner import Tuner
from nni.utils import OptimizeMode, extract_scalar_reward, split_index, json2parameter, json2space


logger = logging.getLogger('pbt_tuner_AutoML')


def exploit_and_explore(bot_trial_info, top_trial_info, factors, epoch, search_space):
    """
    Replace checkpoint of bot_trial with top, and perturb hyperparameters

    Parameters
    ----------
    bot_trial_info : TrialInfo
        bottom model whose parameters should be replaced
    top_trial_info : TrialInfo
        better model
    factors : float
        factors for perturbation
    epoch : int
        step of PBTTuner
    search_space : dict
        search_space to keep perturbed hyperparameters in range
    """
    bot_checkpoint_dir = bot_trial_info.checkpoint_dir
    top_hyper_parameters = top_trial_info.hyper_parameters
    hyper_parameters = copy.deepcopy(top_hyper_parameters)
    # TODO think about different type of hyperparameters for 1.perturbation 2.within search space
    for key in hyper_parameters.keys():
        if key == 'load_checkpoint_dir':
            hyper_parameters[key] = hyper_parameters['save_checkpoint_dir']
        elif key == 'save_checkpoint_dir':
            hyper_parameters[key] = os.path.join(bot_checkpoint_dir, str(epoch))
        elif isinstance(hyper_parameters[key], float):
            perturb = np.random.choice(factors)
            val = hyper_parameters[key] * perturb
            lb, ub = search_space[key]["_value"][:2]
            if search_space[key]["_type"] in ("uniform", "normal"):
                val = np.clip(val, lb, ub).item()
                hyper_parameters[key] = val
        else:
            continue
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


class PBTTuner(Tuner):
    def __init__(self, optimize_mode="maximize", all_checkpoint_dir=None, population_size=10, factors=(1.2, 0.8), fraction=0.2):
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
        factors : tuple
            factors for perturbation
        fraction : float
            fraction for selecting bottom and top trials
        """
        self.optimize_mode = OptimizeMode(optimize_mode)
        if all_checkpoint_dir is None:
            all_checkpoint_dir = os.getenv('NNI_CHECKPOINT_DIRECTORY')
            logger.info("Checkpoint dir is set to %s by default.", all_checkpoint_dir)
        self.all_checkpoint_dir = all_checkpoint_dir
        self.population_size = population_size
        self.factors = factors
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
        return split_index(trial_info.hyper_parameters)

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
        if self.optimize_mode == OptimizeMode.Minimize:
            value = -value
        trial_info = self.running.pop(parameter_id, None)
        trial_info.score = value
        self.finished.append(trial_info)
        self.finished_trials += 1
        if self.finished_trials == self.population_size:
            logger.info('Proceeding to next epoch')
            self.epoch += 1
            self.population = []
            self.pos = -1
            self.running = {}
            #exploit and explore
            self.finished = sorted(self.finished, key=lambda x: x.score, reverse=True)
            cutoff = int(np.ceil(self.fraction * len(self.finished)))
            tops = self.finished[:cutoff]
            bottoms = self.finished[self.finished_trials - cutoff:]
            for bottom in bottoms:
                top = np.random.choice(tops)
                exploit_and_explore(bottom, top, self.factors, self.epoch, self.searchspace_json)
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
                self.send_trial_callback(parameter_id, split_index(trial_info.hyper_parameters))

    def import_data(self, data):
        pass
