# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import copy
import json
import logging
import random
import os

from threading import Event, Lock, current_thread

from nni.tuner import Tuner
from nni.utils import extract_scalar_reward

from graph import Graph, Layer, LayerType, Enum, graph_dumps, graph_loads, unique

logger = logging.getLogger('ga_customer_tuner')


@unique
class OptimizeMode(Enum):
    Minimize = 'minimize'
    Maximize = 'maximize'




class Individual(object):
    """
    Basic Unit for evolution algorithm
    """
    def __init__(self, graph_cfg: Graph = None, info=None, result=None, indiv_id=None):
        self.config = graph_cfg
        self.result = result
        self.info = info
        self.indiv_id = indiv_id
        self.parent_id = None
        self.shared_ids = {layer.hash_id for layer in self.config.layers if layer.is_delete is False}

    def __str__(self):
        return "info: " + str(self.info) + ", config :" + str(self.config) + ", result: " + str(self.result)

    def mutation(self, indiv_id: int, graph_cfg: Graph = None, info=None):
        self.result = None
        if graph_cfg is not None:
            self.config = graph_cfg
        self.config.mutation()
        self.info = info
        self.parent_id = self.indiv_id
        self.indiv_id = indiv_id
        self.shared_ids.intersection_update({layer.hash_id for layer in self.config.layers if layer.is_delete is False})


class CustomerTuner(Tuner):
    """
    NAS Tuner using Evolution Algorithm, with weight sharing enabled
    """
    def __init__(self, optimize_mode, save_dir_root, population_size=32, graph_max_layer=6, graph_min_layer=3):
        self.optimize_mode = OptimizeMode(optimize_mode)
        self.indiv_counter = 0
        self.events = []
        self.thread_lock = Lock()
        self.save_dir_root = save_dir_root
        self.population = self.init_population(population_size, graph_max_layer, graph_min_layer)
        assert len(self.population) == population_size
        logger.debug('init population done.')
        return

    def generate_new_id(self):
        """
        generate new id and event hook for new Individual
        """
        self.events.append(Event())
        indiv_id = self.indiv_counter
        self.indiv_counter += 1
        return indiv_id

    def save_dir(self, indiv_id):
        if indiv_id is None:
            return None
        else:
            return os.path.join(self.save_dir_root, str(indiv_id))

    def init_population(self, population_size, graph_max_layer, graph_min_layer):
        """
        initialize populations for evolution tuner
        """
        population = []
        graph = Graph(max_layer_num=graph_max_layer, min_layer_num=graph_min_layer,
                      inputs=[Layer(LayerType.input.value, output=[4, 5], size='x'), Layer(LayerType.input.value, output=[4, 5], size='y')],
                      output=[Layer(LayerType.output.value, inputs=[4], size='x'), Layer(LayerType.output.value, inputs=[5], size='y')],
                      hide=[Layer(LayerType.attention.value, inputs=[0, 1], output=[2]),
                            Layer(LayerType.attention.value, inputs=[1, 0], output=[3])])
        for _ in range(population_size):
            graph_tmp = copy.deepcopy(graph)
            graph_tmp.mutation()
            population.append(Individual(indiv_id=self.generate_new_id(), graph_cfg=graph_tmp, result=None))
        return population

    def generate_parameters(self, parameter_id, **kwargs):
        """Returns a set of trial graph config, as a serializable object.
        An example configuration:
        ```json
        {
            "shared_id": [
                "4a11b2ef9cb7211590dfe81039b27670",
                "370af04de24985e5ea5b3d72b12644c9",
                "11f646e9f650f5f3fedc12b6349ec60f",
                "0604e5350b9c734dd2d770ee877cfb26",
                "6dbeb8b022083396acb721267335f228",
                "ba55380d6c84f5caeb87155d1c5fa654"
            ],
            "graph": {
                "layers": [
                    ...
                    {
                        "hash_id": "ba55380d6c84f5caeb87155d1c5fa654",
                        "is_delete": false,
                        "size": "x",
                        "graph_type": 0,
                        "output": [
                            6
                        ],
                        "output_size": 1,
                        "input": [
                            7,
                            1
                        ],
                        "input_size": 2
                    },
                    ...
                ]
            },
            "restore_dir": "/mnt/nfs/nni/ga_squad/87",
            "save_dir": "/mnt/nfs/nni/ga_squad/95"
        }
        ```
        `restore_dir` means the path in which to load the previous trained model weights. if null, init from stratch.
        `save_dir` means the path to save trained model for current trial.
        `graph` is the configuration of model network.
                Note: each configuration of layers has a `hash_id` property,
                which tells tuner & trial code whether to share trained weights or not.
        `shared_id` is the hash_id of layers that should be shared with previously trained model.
        """
        logger.debug('acquiring lock for param {}'.format(parameter_id))
        self.thread_lock.acquire()
        logger.debug('lock for current thread acquired')
        if not self.population:
            logger.debug("the len of poplution lower than zero.")
            raise Exception('The population is empty')
        pos = -1
        for i in range(len(self.population)):
            if self.population[i].result is None:
                pos = i
                break
        if pos != -1:
            indiv = copy.deepcopy(self.population[pos])
            self.population.pop(pos)
            graph_param = json.loads(graph_dumps(indiv.config))
        else:
            random.shuffle(self.population)
            if self.population[0].result < self.population[1].result:
                self.population[0] = self.population[1]
            indiv = copy.deepcopy(self.population[0])
            self.population.pop(1)
            indiv.mutation(indiv_id = self.generate_new_id())
            graph_param = json.loads(graph_dumps(indiv.config))
        param_json = {
            'graph': graph_param,
            'restore_dir': self.save_dir(indiv.parent_id),
            'save_dir': self.save_dir(indiv.indiv_id),
            'shared_id': list(indiv.shared_ids) if indiv.parent_id is not None else None,
        }
        logger.debug('generate_parameter return value is:')
        logger.debug(param_json)
        logger.debug('releasing lock')
        self.thread_lock.release()
        if indiv.parent_id is not None:
            logger.debug("new trial {} pending on parent experiment {}".format(indiv.indiv_id, indiv.parent_id))
            self.events[indiv.parent_id].wait()
        logger.debug("trial {} ready".format(indiv.indiv_id))
        return param_json

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        '''
        Record an observation of the objective function
        parameter_id : int
        parameters : dict of parameters
        value: final metrics of the trial, including reward
        '''
        logger.debug('acquiring lock for param {}'.format(parameter_id))
        self.thread_lock.acquire()
        logger.debug('lock for current acquired')
        reward = extract_scalar_reward(value)
        if self.optimize_mode is OptimizeMode.Minimize:
            reward = -reward

        logger.debug('receive trial result is:\n')
        logger.debug(str(parameters))
        logger.debug(str(reward))

        indiv = Individual(indiv_id=int(os.path.split(parameters['save_dir'])[1]),
                           graph_cfg=graph_loads(parameters['graph']), result=reward)
        self.population.append(indiv)
        logger.debug('releasing lock')
        self.thread_lock.release()
        self.events[indiv.indiv_id].set()

    def update_search_space(self, data):
        pass
