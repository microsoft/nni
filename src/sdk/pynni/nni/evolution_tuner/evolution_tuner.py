# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
evolution_tuner.py
"""

import copy
import random
import logging

from collections import deque
import numpy as np
from schema import Schema, Optional

import nni
from nni import ClassArgsValidator
from nni.tuner import Tuner
from nni.utils import OptimizeMode, extract_scalar_reward, split_index, json2parameter, json2space

logger = logging.getLogger(__name__)

class Individual:
    """
    Indicidual class to store the indv info.

    Attributes
    ----------
    config : str
        Search space.
    info : str
        The str to save information of individual.
    result : float
        The final metric of a individual.
    """

    def __init__(self, config=None, info=None, result=None):
        """
        Parameters
        ----------
        config : str
            A config to represent a group of parameters.
        info : str
        result : float
        save_dir : str
        """
        self.config = config
        self.result = result
        self.info = info

    def __str__(self):
        return "info: " + str(self.info) + \
            ", config :" + str(self.config) + ", result: " + str(self.result)

class EvolutionClassArgsValidator(ClassArgsValidator):
    def validate_class_args(self, **kwargs):
        Schema({
            'optimize_mode': self.choices('optimize_mode', 'maximize', 'minimize'),
            Optional('population_size'): self.range('population_size', int, 0, 99999),
        }).validate(kwargs)

class EvolutionTuner(Tuner):
    """
    EvolutionTuner is tuner using navie evolution algorithm.
    """

    def __init__(self, optimize_mode="maximize", population_size=32):
        """
        Parameters
        ----------
        optimize_mode : str, default 'maximize'
        population_size : int
            initial population size. The larger population size,
        the better evolution performance.
        """
        self.optimize_mode = OptimizeMode(optimize_mode)
        self.population_size = population_size

        self.searchspace_json = None
        self.running_trials = {}
        self.num_running_trials = 0
        self.random_state = None
        self.population = None
        self.space = None
        self.credit = 0 # record the unsatisfied trial requests
        self.send_trial_callback = None
        self.param_ids = deque()

    def update_search_space(self, search_space):
        """
        Update search space.

        Search_space contains the information that user pre-defined.

        Parameters
        ----------
        search_space : dict
        """
        self.searchspace_json = search_space
        self.space = json2space(self.searchspace_json)

        self.random_state = np.random.RandomState()
        self.population = []

        for _ in range(self.population_size):
            self._random_generate_individual()

    def trial_end(self, parameter_id, success, **kwargs):
        """
        To deal with trial failure. If a trial fails,
        random generate the parameters and add into the population.
        Parameters
        ----------
        parameter_id : int
            Unique identifier for hyper-parameters used by this trial.
        success : bool
            True if the trial successfully completed; False if failed or terminated.
        **kwargs
            Not used
        """
        self.num_running_trials -= 1
        logger.info('trial (%d) end', parameter_id)

        if not success:
            self.running_trials.pop(parameter_id)
            self._random_generate_individual()

        if self.credit > 1:
            param_id = self.param_ids.popleft()
            config = self._generate_individual(param_id)
            logger.debug('Send new trial (%d, %s) for reducing credit', param_id, config)
            self.send_trial_callback(param_id, config)
            self.credit -= 1
            self.num_running_trials += 1

    def generate_multiple_parameters(self, parameter_id_list, **kwargs):
        """
        Returns multiple sets of trial (hyper-)parameters, as iterable of serializable objects.
        Parameters
        ----------
        parameter_id_list : list of int
            Unique identifiers for each set of requested hyper-parameters.
        **kwargs
            Not used
        Returns
        -------
        list
            A list of newly generated configurations
        """

        result = []
        if 'st_callback' in kwargs:
            self.send_trial_callback = kwargs['st_callback']
        else:
            logger.warning('Send trial callback is not found in kwargs. Evolution tuner might not work properly.')
        for parameter_id in parameter_id_list:
            had_exception = False
            try:
                logger.debug("generating param for %s", parameter_id)
                res = self.generate_parameters(parameter_id, **kwargs)
                self.num_running_trials += 1
            except nni.NoMoreTrialError:
                had_exception = True
            if not had_exception:
                result.append(res)
        return result

    def _random_generate_individual(self):
        is_rand = dict()
        for item in self.space:
            is_rand[item] = True

        config = json2parameter(self.searchspace_json, is_rand, self.random_state)
        self.population.append(Individual(config=config))

    def _generate_individual(self, parameter_id):
        """
        This function will generate the config for a trial.
        If at the first generation, randomly generates individuals to satisfy self.population_size.
        Otherwise, random choose a pair of individuals and compare their fitnesses.
        The worst of the pair will be removed. Copy the best of the pair and mutate it to generate a new individual.

        Parameters
        ----------
        parameter_id : int

        Returns
        -------
        dict
            A group of candaidte parameters that evolution tuner generated.
        """
        pos = -1

        for i in range(len(self.population)):
            if self.population[i].result is None:
                pos = i
                break

        if pos != -1:
            indiv = copy.deepcopy(self.population[pos])
            self.population.pop(pos)
        else:
            random.shuffle(self.population)
            # avoid only 1 individual has result
            if len(self.population) > 1 and self.population[0].result < self.population[1].result:
                self.population[0] = self.population[1]

            # mutation on the worse individual
            space = json2space(self.searchspace_json,
                               self.population[0].config)
            is_rand = dict()
            mutation_pos = space[random.randint(0, len(space)-1)]

            for i in range(len(self.space)):
                is_rand[self.space[i]] = (self.space[i] == mutation_pos)
            config = json2parameter(
                self.searchspace_json, is_rand, self.random_state, self.population[0].config)

            if len(self.population) > 1:
                self.population.pop(1)

            indiv = Individual(config=config)

        # remove "_index" from config and save params-id
        self.running_trials[parameter_id] = indiv
        config = split_index(indiv.config)
        return config


    def generate_parameters(self, parameter_id, **kwargs):
        """
        This function will returns a dict of trial (hyper-)parameters.
        If no trial configration for now, self.credit plus 1 to send the config later

        Parameters
        ----------
        parameter_id : int

        Returns
        -------
        dict
            One newly generated configuration.
        """
        if not self.population:
            raise RuntimeError('The population is empty')

        if self.num_running_trials >= self.population_size:
            logger.warning("No enough trial config, population_size is suggested to be larger than trialConcurrency")
            self.credit += 1
            self.param_ids.append(parameter_id)
            raise nni.NoMoreTrialError('no more parameters now.')

        return self._generate_individual(parameter_id)

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        """
        Record the result from a trial

        Parameters
        ----------
        parameter_id : int
        parameters : dict
        value : dict/float
            if value is dict, it should have "default" key.
            value is final metrics of the trial.
        """
        reward = extract_scalar_reward(value)

        if parameter_id not in self.running_trials:
            raise RuntimeError('Received parameter_id %s not in running_trials.', parameter_id)

        # restore the paramsters contains "_index"
        config = self.running_trials[parameter_id].config
        self.running_trials.pop(parameter_id)

        if self.optimize_mode == OptimizeMode.Minimize:
            reward = -reward

        indiv = Individual(config=config, result=reward)
        self.population.append(indiv)

    def import_data(self, data):
        pass
