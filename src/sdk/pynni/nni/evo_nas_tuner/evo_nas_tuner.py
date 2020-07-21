# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
evo_nas_tuner.py
"""

import copy
import random
import logging

import numpy as np
from schema import Schema, Optional

from nni import ClassArgsValidator
from nni.tuner import Tuner
from nni.utils import OptimizeMode, extract_scalar_reward, split_index, json2parameter, json2space

logger = logging.getLogger(__name__)

class FinishedIndividual:
    def __init__(self, parameter_id, parameters, result):
        """
        Parameters
        ----------
        parameters : dict
            chosen architecture and parameters
        result : float
            final metric of the chosen one
        """
        self.parameter_id = parameter_id
        self.parameters = parameters
        self.result = result

class EvoNasTuner(Tuner):
    def __init__(self, optimize_mode="maximize", population_size=100, sample_size=10, wait_initial=False):
        """
        Parameters
        ----------
        population_size : int
            population size, fixed during the whole search process
        wait_initial : bool
            whether to wait for the initial population to finish, then create new individuals
        """
        self.optimize_mode = OptimizeMode(optimize_mode)
        self.population_size = population_size
        self.sample_size = sample_size
        self.initial_population = [] # element is arch
        self.population = [] # element is finished individual
        self.search_space = None
        self.chosen_arch_template = {}
        self.total_data = {}
        self.send_trial_callback = None

    def _randomize_individual(self):
        individual = {}
        for key, val in self.search_space.items():
            if val['_type'] == 'layer_choice':
                idx = random.randint(0, len(val['_value']) - 1)
                individual[key] = { '_value': val['_value'][idx], '_idx':  idx }
            elif val['_type'] == 'input_choice':
                candidates = val['_value']['candidates']
                # TODO: deal with the value None
                n_chosen = val['_value']['n_chosen']
                idxs = [ random.randint(0, len(candidates) - 1) for _ in range(n_chosen) ]
                vals = [ candidates[k] for k in idxs ]
                individual[key] = { '_value': vals, '_idx': idxs }
        return individual

    def _generate_initial_population(self):
        for _ in range(self.population_size):
            individual = self._randomize_individual()
            self.initial_population.append(individual)

    def update_search_space(self, search_space):
        """
        Get search space, currently the space only includes that for NAS

        Parameters
        ----------
        search_space : dict
            Search space for NAS
            the format ...
        """
        logger.info('update search space %s', search_space)
        assert self.search_space is None
        self.search_space = search_space

        for key, val in search_space.items():
            if val['_type'] != 'layer_choice' and val['_type'] != 'input_choice':
                raise ValueError('Unsupported search space type: %s' % (val['_type']))

        self._generate_initial_population()

    def _get_best(self, samples):
        best_sample = None
        val = float('-inf')
        for sample in samples:
            if val < sample.result:
                val = sample.result
                best_sample = sample
        return best_sample

    def _mutate_individual(self, individual):
        """
        Parameters
        ----------
        individual : FinishedIndividual
            parent

        Returns
        -------
        dict
            the newly mutated one
        """
        new_individual = copy.deepcopy(individual.parameters)
        keys = [ key for key in individual.parameters ]
        mutate_key = random.choice(keys)
        val = self.search_space[mutate_key]
        if val['_type'] == 'layer_choice':
            idx = random.randint(0, len(val['_value']) - 1)
            new_individual[mutate_key] = { '_value': val['_value'][idx], '_idx':  idx }
        elif self.search_space[mutate_key]['_type'] == 'input_choice':
            candidates = val['_value']['candidates']
            n_chosen = val['_value']['n_chosen']
            idxs = [ random.randint(0, len(candidates) - 1) for _ in range(n_chosen) ]
            vals = [ candidates[k] for k in idxs ]
            new_individual[mutate_key] = { '_value': vals, '_idx': idxs }
        else:
            raise
        return new_individual

    def _mutate_population(self):
        """
        """
        samples = []
        for _ in range(self.sample_size):
            chosen_idx = random.randint(0, len(self.population) - 1)
            samples.append(self.population[chosen_idx])
        parent = self._get_best(samples)
        return self._mutate_individual(parent)

    def generate_parameters(self, parameter_id, **kwargs):
        """
        """
        if self.initial_population:
            individual =  self.initial_population.pop()
            self.total_data[parameter_id] = individual
            return individual
        elif self.population:
            individual = self._mutate_population()
            self.total_data[parameter_id] = individual
            return individual
        else:
            # TODO: reconsider this part, randomly generate one here?
            raise nni.NoMoreTrialError

    def generate_multiple_parameters(self, parameter_id_list, **kwargs):
        """
        Returns multiple sets of trial (hyper-)parameters, as iterable of serializable objects.

        Parameters
        ----------
        parameter_id_list : list of int
            Unique identifiers for each set of requested hyper-parameters.
            These will later be used in :meth:`receive_trial_result`.
        **kwargs
            Not used

        Returns
        -------
        list
            A list of newly generated configurations
        """
        if len(parameter_id_list) > self.population_size:
            logger.warning('trialConcurrency {} is larger than population size {}, resource cannot be fully utilized'.format(
                           len(parameter_id_list), self.population_size))
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
        params = self.total_data[parameter_id]

        if self.optimize_mode == OptimizeMode.Minimize:
            reward = -reward
        
        self.population.append(FinishedIndividual(parameter_id, params, reward))
        if len(self.population) > self.population_size:
            self.population.pop()

    def trial_end(self, parameter_id, success, **kwargs):
        """
        """
        pass

    def import_data(self, data):
        """
        Import additional data for tuning, not supported yet.

        Parameters
        ----------
        data : list
            A list of dictionarys, each of which has at least two keys, ``parameter`` and ``value``
        """
        logger.warning('pending')
