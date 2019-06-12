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
"""
evolution_tuner.py
"""

import copy
import random

import numpy as np
from nni.tuner import Tuner
from nni.utils import NodeType, OptimizeMode, extract_scalar_reward, split_index, randint_to_quniform

import nni.parameter_expressions as parameter_expressions


def json2space(x, oldy=None, name=NodeType.ROOT):
    """Change search space from json format to hyperopt format
    """
    y = list()
    if isinstance(x, dict):
        if NodeType.TYPE in x.keys():
            _type = x[NodeType.TYPE]
            name = name + '-' + _type
            if _type == 'choice':
                if oldy != None:
                    _index = oldy[NodeType.INDEX]
                    y += json2space(x[NodeType.VALUE][_index],
                                    oldy[NodeType.VALUE], name=name+'[%d]' % _index)
                else:
                    y += json2space(x[NodeType.VALUE], None, name=name)
            y.append(name)
        else:
            for key in x.keys():
                y += json2space(x[key], (oldy[key] if oldy !=
                                         None else None), name+"[%s]" % str(key))
    elif isinstance(x, list):
        for i, x_i in enumerate(x):
            if isinstance(x_i, dict):
                if NodeType.NAME not in x_i.keys():
                    raise RuntimeError('\'_name\' key is not found in this nested search space.')
            y += json2space(x_i, (oldy[i] if oldy !=
                                  None else None), name+"[%d]" % i)
    return y

def json2parameter(x, is_rand, random_state, oldy=None, Rand=False, name=NodeType.ROOT):
    """Json to pramaters.
    """
    if isinstance(x, dict):
        if NodeType.TYPE in x.keys():
            _type = x[NodeType.TYPE]
            _value = x[NodeType.VALUE]
            name = name + '-' + _type
            Rand |= is_rand[name]
            if Rand is True:
                if _type == 'choice':
                    _index = random_state.randint(len(_value))
                    y = {
                        NodeType.INDEX: _index,
                        NodeType.VALUE: json2parameter(x[NodeType.VALUE][_index],
                                                             is_rand,
                                                             random_state,
                                                             None,
                                                             Rand,
                                                             name=name+"[%d]" % _index)
                    }
                else:
                    y = eval('parameter_expressions.' +
                             _type)(*(_value + [random_state]))
            else:
                y = copy.deepcopy(oldy)
        else:
            y = dict()
            for key in x.keys():
                y[key] = json2parameter(x[key], is_rand, random_state, oldy[key]
                                        if oldy != None else None, Rand, name + "[%s]" % str(key))
    elif isinstance(x, list):
        y = list()
        for i, x_i in enumerate(x):
            if isinstance(x_i, dict):
                if NodeType.NAME not in x_i.keys():
                    raise RuntimeError('\'_name\' key is not found in this nested search space.')
            y.append(json2parameter(x_i, is_rand, random_state, oldy[i]
                                    if oldy != None else None, Rand, name + "[%d]" % i))
    else:
        y = copy.deepcopy(x)
    return y

class Individual(object):
    """
    Indicidual class to store the indv info.
    """

    def __init__(self, config=None, info=None, result=None, save_dir=None):
        """
        Parameters
        ----------
        config : str
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


class EvolutionTuner(Tuner):
    """
    EvolutionTuner is tuner using navie evolution algorithm.
    """

    def __init__(self, optimize_mode, population_size=32):
        """
        Parameters
        ----------
        optimize_mode : str
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
        """Update search space. 
        Search_space contains the information that user pre-defined.

        Parameters
        ----------
        search_space : dict
        """
        self.searchspace_json = search_space
        randint_to_quniform(self.searchspace_json)
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

    def generate_parameters(self, parameter_id):
        """Returns a dict of trial (hyper-)parameters, as a serializable object.

        Parameters
        ----------
        parameter_id : int
    
        Returns
        -------
        config : dict
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

    def receive_trial_result(self, parameter_id, parameters, value):
        '''Record the result from a trial

        Parameters
        ----------
        parameters: dict
        value : dict/float
            if value is dict, it should have "default" key.
            value is final metrics of the trial.
        '''
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
