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


import copy
import logging

from enum import Enum, unique
import numpy as np

from nni.tuner import Tuner

logger = logging.getLogger('NetworkMorphism_AutoML')



class NetworkMorphismTuner(Tuner):
    '''
    NetworkMorphismTuner is a tuner which using baysain algorithm.
    '''
    
    def __init__(self, algorithm_name, optimize_mode):
        self.algorithm_name = algorithm_name
        self.optimize_mode = optimize_mode
        self.json = None
        self.total_data = {}


    def update_search_space(self, search_space):
        '''
        Update search space definition in tuner by search_space in parameters.
        '''
        pass

    def generate_parameters(self, parameter_id):
        '''
        Returns a set of trial (hyper-)parameters, as a serializable object.
        parameter_id : int
        '''
        params = {}
        return params

    def receive_trial_result(self, parameter_id, parameters, value):
        '''
        Record an observation of the objective function
        parameter_id : int
        parameters : dict of parameters
        value: final metrics of the trial, including reward
        '''
        pass


