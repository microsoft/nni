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
gridsearch_tuner.py including:
    class GridSearchTuner
'''

import copy
import numpy as np
import logging

import nni
from nni.tuner import Tuner
from nni.utils import convert_dict2tuple

TYPE = '_type'
CHOICE = 'choice'
VALUE = '_value'

logger = logging.getLogger('grid_search_AutoML')

class GridSearchTuner(Tuner):
    '''
    GridSearchTuner will search all the possible configures that the user define in the searchSpace.
    The only acceptable types of search space are 'quniform', 'qloguniform' and 'choice'

    Type 'choice' will select one of the options. Note that it can also be nested.

    Type 'quniform' will receive three values [low, high, q], where [low, high] specifies a range and 'q' specifies the number of values that will be sampled evenly.
    Note that q should be at least 2.
    It will be sampled in a way that the first sampled value is 'low', and each of the following values is (high-low)/q larger that the value in front of it.

    Type 'qloguniform' behaves like 'quniform' except that it will first change the range to [log(low), log(high)]
    and sample and then change the sampled value back.
    '''

    def __init__(self):
        self.count = -1
        self.expanded_search_space = []
        self.supplement_data = dict()

    def json2parameter(self, ss_spec):
        '''
        generate all possible configs for hyperparameters from hyperparameter space.
        ss_spec: hyperparameter space
        '''
        if isinstance(ss_spec, dict):
            if '_type' in ss_spec.keys():
                _type = ss_spec['_type']
                _value = ss_spec['_value']
                chosen_params = list()
                if _type == 'choice':
                    for value in _value:
                        choice = self.json2parameter(value)
                        if isinstance(choice, list):
                            chosen_params.extend(choice)
                        else:
                            chosen_params.append(choice)
                else:
                    chosen_params = self.parse_qtype(_type, _value)
            else:
                chosen_params = dict()
                for key in ss_spec.keys():
                    chosen_params[key] = self.json2parameter(ss_spec[key])
                return self.expand_parameters(chosen_params)
        elif isinstance(ss_spec, list):
            chosen_params = list()
            for subspec in ss_spec[1:]:
                choice = self.json2parameter(subspec)
                if isinstance(choice, list):
                    chosen_params.extend(choice)
                else:
                    chosen_params.append(choice)
            chosen_params = list(map(lambda v: {ss_spec[0]: v}, chosen_params))
        else:
            chosen_params = copy.deepcopy(ss_spec)
        return chosen_params

    def _parse_quniform(self, param_value):
        '''parse type of quniform parameter and return a list'''
        if param_value[2] < 2:
            raise RuntimeError("The number of values sampled (q) should be at least 2")
        low, high, count = param_value[0], param_value[1], param_value[2]
        interval = (high - low) / (count - 1)
        return [float(low + interval * i) for i in range(count)]

    def parse_qtype(self, param_type, param_value):
        '''parse type of quniform or qloguniform'''
        if param_type == 'quniform':
            return self._parse_quniform(param_value)
        if param_type == 'qloguniform':
            param_value[:2] = np.log(param_value[:2])
            return list(np.exp(self._parse_quniform(param_value)))

        raise RuntimeError("Not supported type: %s" % param_type)

    def expand_parameters(self, para):
        '''
        Enumerate all possible combinations of all parameters
        para: {key1: [v11, v12, ...], key2: [v21, v22, ...], ...}
        return: {{key1: v11, key2: v21, ...}, {key1: v11, key2: v22, ...}, ...}
        '''
        if len(para) == 1:
            for key, values in para.items():
                return list(map(lambda v: {key: v}, values))

        key = list(para)[0]
        values = para.pop(key)
        rest_para = self.expand_parameters(para)
        ret_para = list()
        for val in values:
            for config in rest_para:
                config[key] = val
                ret_para.append(copy.deepcopy(config))
        return ret_para

    def update_search_space(self, search_space):
        '''
        Check if the search space is valid and expand it: only contains 'choice' type or other types beginnning with the letter 'q'
        '''
        self.expanded_search_space = self.json2parameter(search_space)

    def generate_parameters(self, parameter_id):
        self.count += 1
        while (self.count <= len(self.expanded_search_space)-1):
            _params_tuple = convert_dict2tuple(self.expanded_search_space[self.count])
            if _params_tuple in self.supplement_data:
                self.count += 1
            else:
                return self.expanded_search_space[self.count]
        raise nni.NoMoreTrialError('no more parameters now.')

    def receive_trial_result(self, parameter_id, parameters, value):
        pass

    def import_data(self, data):
        """Import additional data for tuning

        Parameters
        ----------
        data:
            a list of dictionarys, each of which has at least two keys, 'parameter' and 'value'
        """
        _completed_num = 0
        for trial_info in data:
            logger.info("Importing data, current processing progress %s / %s" %(_completed_num, len(data)))
            _completed_num += 1
            assert "parameter" in trial_info
            _params = trial_info["parameter"]
            assert "value" in trial_info
            _value = trial_info['value']
            if not _value:
                logger.info("Useless trial data, value is %s, skip this trial data." %_value)
                continue
            _params_tuple = convert_dict2tuple(_params)
            self.supplement_data[_params_tuple] = True
        logger.info("Successfully import data to grid search tuner.")
