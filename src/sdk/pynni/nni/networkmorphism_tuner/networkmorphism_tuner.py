# Copyright (c) Microsoft Corporation. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
# OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ==================================================================================================


import copy
import logging
import os
import re
import time
from enum import Enum, unique

import numpy as np

from nni.tuner import Tuner

from nni.networkmorphism_tuner.bayesian import BayesianOptimizer
from nni.networkmorphism_tuner.metric import Accuracy
from nni.networkmorphism_tuner.utils import pickle_to_file, pickle_from_file, Constant
from nni.networkmorphism_tuner.net_transformer import default_transform
from nni.networkmorphism_tuner.nn import CnnGenerator
from nni.networkmorphism_tuner.graph import onnx_to_graph
import onnx
import torch

logger = logging.getLogger('NetworkMorphism_AutoML')


@unique
class OptimizeMode(Enum):
    '''
    Oprimize Mode class
    '''
    Minimize = 'minimize'
    Maximize = 'maximize'


class SearchTree:
    def __init__(self):
        self.root = None
        self.adj_list = {}

    def add_child(self, u, v):
        if u == -1:
            self.root = v
            self.adj_list[v] = []
            return
        if v not in self.adj_list[u]:
            self.adj_list[u].append(v)
        if v not in self.adj_list:
            self.adj_list[v] = []

    def get_dict(self, u=None):
        if u is None:
            return self.get_dict(self.root)
        children = []
        for v in self.adj_list[u]:
            children.append(self.get_dict(v))
        ret = {'name': u, 'children': children}
        return ret


class NetworkMorphismTuner(Tuner):
    '''
    NetworkMorphismTuner is a tuner which using network morphism techniques.
    '''

    def __init__(self, input_shape=(32,32,3), n_output_node=10, algorithm_name="Bayesian",optimize_mode="minimize", path=None, verbose=True, metric=Accuracy, beta=Constant.BETA,
                 kernel_lambda=Constant.KERNEL_LAMBDA, t_min=None, max_model_size=Constant.MAX_MODEL_SIZE, default_model_len=Constant.MODEL_LEN, default_model_width=Constant.MODEL_WIDTH):
        if path is None:
            os.makedirs("model_path")
            path = "model_path"
        else:
            if not os.path.exists(path):
                os.makedirs(path)
        self.path = path
        self.n_classes = n_output_node
        self.input_shape = input_shape

        self.t_min = t_min
        self.metric = metric
        self.kernel_lambda = kernel_lambda
        self.beta = beta
        self.algorithm_name = algorithm_name
        self.optimize_mode = OptimizeMode(optimize_mode)
        self.json = None
        self.total_data = {}
        self.verbose = verbose
        self.model_count = 0

        self.bo = BayesianOptimizer(self, self.t_min, self.metric, self.kernel_lambda, self.beta)
        self.training_queue = []
        self.x_queue = []
        self.y_queue = []
        self.search_tree = SearchTree()
        self.descriptors = []

        self.max_model_size = max_model_size
        self.default_model_len = default_model_len
        self.default_model_width = default_model_width

        self.search_space = []

    def _choose_tuner(self, algorithm_name):
        ''' choose algorithm of tuner

        Arguments:
            algorithm_name: str -- micro-level, basic operation default in baysain algorithm; meso-level, modularized morphism; macro-level, evolution learning

        Raises:
            RuntimeError -- Not support tuner algorithm in Network Morphism.
        '''
        pass

    def update_search_space(self, search_space):
        '''
        Update search space definition in tuner by search_space in neural architecture.
        '''
        self.search_space = search_space
        

    def generate_parameters(self, parameter_id):
        '''
        Returns a set of trial neural architecture, as a serializable object.
        parameter_id : int
        '''
        if not self.history:
            self.init_search()

        new_graph = None
        new_father_id = None
        if not self.training_queue:
            while new_father_id is None:
                new_graph, new_father_id = self.bo.optimize_acq(
                    self.search_tree.adj_list.keys(), self.descriptors)
            new_model_id = self.model_count
            self.model_count += 1
            self.training_queue.append(
                (new_graph, new_father_id, new_model_id))
            self.descriptors.append(new_graph.extract_descriptor())

        graph, father_id, model_id = self.training_queue.pop(0)

        # from gragh to onnx_model_path
        onnx_model_path = os.path.join(self.path, str(model_id) + '.onnx')
        torch_model = graph.produce_model()
        x = torch.randn(Constant.BATCH_SIZE, self.input_shape[2], self.input_shape[0], self.input_shape[1], requires_grad=True)
        torch_out = torch.onnx.export(torch_model, x , onnx_model_path,export_params=True)    
        # onnx.save(graph, onnx_model_path)

        self.total_data[parameter_id] = (onnx_model_path, father_id, model_id)

        return onnx_model_path

    def receive_trial_result(self, parameter_id, parameters, value):
        ''' Record an observation of the objective function

        Arguments:           
            parameter_id : int
            parameters : dict of parameters
            value: final metrics of the trial, including reward     

        Raises:
            RuntimeError -- Received parameter_id not in total_data.
        '''

        reward = self.extract_scalar_reward(value)

        if parameter_id not in self.total_data:
            raise RuntimeError('Received parameter_id not in total_data.')

        (onnx_model_path, father_id, model_id) = self.total_data[parameter_id]

        if self.optimize_mode is OptimizeMode.Maximize:
            reward = -reward

        # from onnx_model_path to gragh
        onnx_graph = onnx.load(onnx_model_path)
        graph = onnx_to_graph(onnx_graph,self.input_shape)

        # to use the reward and graph
        self.add_model(reward, graph, model_id)
        self.search_tree.add_child(father_id, model_id)

        self.bo.fit(self.x_queue, self.y_queue)
        self.x_queue = []
        self.y_queue = []
        self.history = []

        pickle_to_file(self, os.path.join(self.path, 'searcher'))

    def init_search(self):
        if self.verbose:
            logger.info('Initializing search.')
        graph = CnnGenerator(self.n_classes, self.input_shape).generate(
            self.default_model_len, self.default_model_width)
            
        model_id = self.model_count
        self.model_count += 1
        self.training_queue.append((graph, -1, model_id))
        self.descriptors.append(graph.extract_descriptor())
        for child_graph in default_transform(graph):
            child_id = self.model_count
            self.model_count += 1
            self.training_queue.append((child_graph, model_id, child_id))
            self.descriptors.append(child_graph.extract_descriptor())
        if self.verbose:
            logger.info('Initialization finished.')

    def add_model(self, metric_value, graph, model_id):
        ''' add model to the history, x_queue and y_queue

        Arguments:
            metric_value: int --metric_value
            graph: dict -- graph
            model_id: int -- model_id

        Returns:
            model dict
        '''

        if self.verbose:
            logger.info('Saving model.')

        # Update best_model text file
        ret = {'model_id': model_id, 'metric_value': metric_value}
        self.history.append(ret)
        if model_id == self.get_best_model_id():
            file = open(os.path.join(self.path, 'best_model.txt'), 'w')
            file.write('best model: ' + str(model_id))
            file.close()

        if self.verbose:
            idx = ['model_id', 'metric_value']
            header = ['Model ID', 'Metric Value']
            line = '|'.join(x.center(24) for x in header)
            logger.info('+' + '-' * len(line) + '+')
            logger.info('|' + line + '|')

            if self.history:
                r = self.history[-1]
                logger.info('+' + '-' * len(line) + '+')
                line = '|'.join(str(r[x]).center(24) for x in idx)
                logger.info('|' + line + '|')
            logger.info('+' + '-' * len(line) + '+')

        descriptor = graph.extract_descriptor()
        self.x_queue.append(descriptor)
        self.y_queue.append(metric_value)
        return ret

    def get_best_model_id(self):
        ''' get the best model_id from history using the metric value

        Returns:
            int -- the best model_id
        '''

        if self.metric.higher_better():
            return max(self.history, key=lambda x: x['metric_value'])['model_id']
        return min(self.history, key=lambda x: x['metric_value'])['model_id']

    def load_model_by_id(self, model_id):
        return onnx.load(os.path.join(self.path, str(model_id) + '.onnx'))

    def load_best_model(self):
        return self.load_model_by_id(self.get_best_model_id())
