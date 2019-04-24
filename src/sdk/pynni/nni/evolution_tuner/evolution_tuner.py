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
evolution_tuner.py including:
    class OptimizeMode
    class Individual
    class EvolutionTuner
"""

import copy
from enum import Enum, unique
import random

import numpy as np

from nni.tuner import Tuner
from nni.utils import extract_scalar_reward
from .. import parameter_expressions

@unique
class OptimizeMode(Enum):
    """Optimize Mode class

    if OptimizeMode is 'minimize', it means the tuner need to minimize the reward
    that received from Trial.

    if OptimizeMode is 'maximize', it means the tuner need to maximize the reward
    that received from Trial.
    """
    Minimize = 'minimize'
    Maximize = 'maximize'


@unique
class NodeType(Enum):
    """Node Type class
    """
    Root = 'root'
    Type = '_type'
    Value = '_value'
    Index = '_index'


def json2space(x, oldy=None, name=NodeType.Root.value):
    """Change search space from json format to hyperopt format
    """
    y = list()
    if isinstance(x, dict):
        if NodeType.Type.value in x.keys():
            _type = x[NodeType.Type.value]
            name = name + '-' + _type
            if _type == 'choice':
                if oldy != None:
                    _index = oldy[NodeType.Index.value]
                    y += json2space(x[NodeType.Value.value][_index],
                                    oldy[NodeType.Value.value], name=name+'[%d]' % _index)
                else:
                    y += json2space(x[NodeType.Value.value], None, name=name)
            y.append(name)
        else:
            for key in x.keys():
                y += json2space(x[key], (oldy[key] if oldy !=
                                         None else None), name+"[%s]" % str(key))
    elif isinstance(x, list):
        for i, x_i in enumerate(x):
            y += json2space(x_i, (oldy[i] if oldy !=
                                  None else None), name+"[%d]" % i)
    else:
        pass
    return y


def json2paramater(x, is_rand, random_state, oldy=None, Rand=False, name=NodeType.Root.value):
    """Json to pramaters.
    """
    if isinstance(x, dict):
        if NodeType.Type.value in x.keys():
            _type = x[NodeType.Type.value]
            _value = x[NodeType.Value.value]
            name = name + '-' + _type
            Rand |= is_rand[name]
            if Rand is True:
                if _type == 'choice':
                    _index = random_state.randint(len(_value))
                    y = {
                        NodeType.Index.value: _index,
                        NodeType.Value.value: json2paramater(x[NodeType.Value.value][_index],
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
                y[key] = json2paramater(x[key], is_rand, random_state, oldy[key]
                                        if oldy != None else None, Rand, name + "[%s]" % str(key))
    elif isinstance(x, list):
        y = list()
        for i, x_i in enumerate(x):
            y.append(json2paramater(x_i, is_rand, random_state, oldy[i]
                                    if oldy != None else None, Rand, name + "[%d]" % i))
    else:
        y = copy.deepcopy(x)
    return y


def _split_index(params):
    """Delete index information from params

    Parameters
    ----------
    params : dict

    Returns
    -------
    result : dict
    """
    result = {}
    for key in params:
        if isinstance(params[key], dict):
            value = params[key]['_value']
        else:
            value = params[key]
        result[key] = value
    return result


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
        self.space = json2space(self.searchspace_json)

        self.random_state = np.random.RandomState()
        self.population = []
        is_rand = dict()
        for item in self.space:
            is_rand[item] = True
        for _ in range(self.population_size):
            config = json2paramater(
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
            config = json2paramater(
                self.searchspace_json, is_rand, self.random_state, self.population[0].config)
            self.population.pop(1)
            # remove "_index" from config and save params-id

            total_config = config
        self.total_data[parameter_id] = total_config
        config = _split_index(total_config)
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
