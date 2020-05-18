# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
evolution_tuner.py
"""

import copy
import random

import numpy as np
from schema import Schema, Optional

from nni import ClassArgsValidator
from nni.tuner import Tuner
from nni.utils import OptimizeMode, extract_scalar_reward, split_index, json2parameter, json2space


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
    store_dir : str
    save_dir : str
    """

    def __init__(self, config=None, info=None, result=None, save_dir=None):
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
        self.restore_dir = None
        self.save_dir = save_dir

    def __str__(self):
        return "info: " + str(self.info) + \
            ", config :" + str(self.config) + ", result: " + str(self.result)

    def mutation(self, config=None, info=None, save_dir=None):
        """
        Mutation by reset state information.

        Parameters
        ----------
        config : str
        info : str
        save_dir : str
        """
        self.result = None
        self.config = config
        self.restore_dir = self.save_dir
        self.save_dir = save_dir
        self.info = info

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

        self.trial_result = []
        self.searchspace_json = None
        self.total_data = {}
        self.random_state = None
        self.population = None
        self.space = None


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
        is_rand = dict()

        for item in self.space:
            is_rand[item] = True

        for _ in range(self.population_size):
            config = json2parameter(
                self.searchspace_json, is_rand, self.random_state)
            self.population.append(Individual(config=config))


    def generate_parameters(self, parameter_id, **kwargs):
        """
        This function will returns a dict of trial (hyper-)parameters, as a serializable object.

        Parameters
        ----------
        parameter_id : int

        Returns
        -------
        dict
            A group of candaidte parameters that evolution tuner generated.
        """
        if not self.population:
            raise RuntimeError('The population is empty')

        pos = -1

        for i in range(len(self.population)):
            if self.population[i].result is None:
                pos = i
                break

        if pos != -1:
            indiv = copy.deepcopy(self.population[pos])
            self.population.pop(pos)
            total_config = indiv.config
        else:
            random.shuffle(self.population)
            if self.population[0].result < self.population[1].result:
                self.population[0] = self.population[1]

            # mutation
            space = json2space(self.searchspace_json,
                               self.population[0].config)
            is_rand = dict()
            mutation_pos = space[random.randint(0, len(space)-1)]

            for i in range(len(self.space)):
                is_rand[self.space[i]] = (self.space[i] == mutation_pos)
            config = json2parameter(
                self.searchspace_json, is_rand, self.random_state, self.population[0].config)
            self.population.pop(1)
            # remove "_index" from config and save params-id

            total_config = config

        self.total_data[parameter_id] = total_config
        config = split_index(total_config)

        return config


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

        if parameter_id not in self.total_data:
            raise RuntimeError('Received parameter_id not in total_data.')
        # restore the paramsters contains "_index"
        params = self.total_data[parameter_id]

        if self.optimize_mode == OptimizeMode.Minimize:
            reward = -reward

        indiv = Individual(config=params, result=reward)
        self.population.append(indiv)

    def import_data(self, data):
        pass
