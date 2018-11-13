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
'''
batch_tuner.py including:
    class BatchTuner
'''

import copy
from enum import Enum, unique
import random

import numpy as np

import nni
from nni.tuner import Tuner

TYPE = '_type'
CHOICE = 'choice'
VALUE = '_value'


class GridSearchTuner(Tuner):
    '''
    GridSearchTuner will search all the possible configures that the user define in the searchSpace.
    '''
    
    def __init__(self):
        self.count = -1
        self.expanded_search_space = []

    def parse(self, param_value):
        q = param_value[2]
        func = lambda x:int(round(x/q))
        lower, upper = func(param_value[0]), func(param_value[1])
        return [float(i*q) for i in range(lower, upper+1)]

    def parse_parameter(self, param_type, param_value):
        if param_type in ['quniform', 'qnormal']:
            return self.parse(param_value, lambda v:v)
        elif param_type in ['qloguniform', 'qlognormal']:
            param_value[0] = np.exp(param_value[0])
            param_value[1] = np.exp(param_value[1])
            return self.parse(param_value)
        else:
            raise RuntimeError("Not supported type: %s" % param_type)

    def expand_parameters(self, para):
        if len(para) == 1:
            for key, values in para.items():
                return list(map(lambda v:{key:v}, values))
        
        key = list(para)[0]
        values = para.pop(key)
        rest_para = self.expand_parameters(para)
        ret_para = list()
        for val in values:
            for config in rest_para:
                config[key] = val
                ret_para.append(dict(config))
        return ret_para

    def update_search_space(self, search_space):
        '''
        Check if the search space is valid and expand it: only contains 'choice' type or other types beginnning with the letter 'q'
        '''
        ss = dict()
        for param in search_space:
            param_type = search_space[param][TYPE]
            param_value = search_space[param][VALUE]
            if param_type == CHOICE:
               ss[param] = param_value
            elif param_type[0] == 'q':
                ss[param] = parse_parameter(param_type, param_value)
            else:
                raise RuntimeError("GridSearchTuner only supprt the 'choice' type or other types beginnning with the letter 'q'")

        self.expanded_search_space = self.expand_parameters(ss)

    def generate_parameters(self, parameter_id):
        self.count +=1
        if self.count > len(self.expanded_search_space)-1:
            raise nni.NoMoreTrialError('no more parameters now.')
        return self.expanded_search_space[self.count]

    def receive_trial_result(self, parameter_id, parameters, value):
        pass