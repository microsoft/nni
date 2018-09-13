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

from nni.tuner import Tuner
from . import parameter_expressions


TYPE = '_type'
CHOICE = 'choice'
VALUE = '_value'


class BatchTuner(Tuner):
    '''
    BatchTuner is tuner will running all the configure that user want to run batchly.
    The search space only be accepted like:
    {
        'combine_params': { '_type': 'choice',
                             '_value': '[{...}, {...}, {...}]',
                          }
    }
    '''
    
    def __init__(self):
        self.count = -1
        self.values = []

    def is_valid(self, search_space)
        '''
        Check the search space is valid: only contains 'choice' type
        '''
        if not len(search_space) == 1:
            raise RuntimeException('BatchTuner only supprt one combined-paramreters key.')
        
        for param in search_space:
            param_type = param[TYPE]
            if param_type is not CHOICE:
                raise RuntimeException('BatchTuner only supprt one combined-paramreters type is choice.')
            else:
                if isinstance(param[VALUE], list):
                    return param[VALUE]
                raise RuntimeException('The combined-paramreters value in BatchTuner is not a list.')
        return None

    def update_search_space(self, search_space):
        self.values = is_valid(search_space)

    def generate_parameters(self, parameter_id):
        count +=1
        if count>len(self.value)-1:
            return None
        return self.values[count]

    def receive_trial_result(self, parameter_id, parameters, reward):
        pass