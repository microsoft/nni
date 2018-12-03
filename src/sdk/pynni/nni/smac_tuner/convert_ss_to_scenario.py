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

import os
import json
import numpy as np

def get_json_content(file_path):
    '''Load json file content'''
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except TypeError as err:
        print('Error: ', err)
        return None

def generate_pcs(nni_search_space_content):
    '''
    # parameter_name categorical {value_1, ..., value_N} [default value]
    # parameter_name ordinal {value_1, ..., value_N} [default value]
    # parameter_name integer [min_value, max_value] [default value]
    # parameter_name integer [min_value, max_value] [default value] log
    # parameter_name real [min_value, max_value] [default value]
    # parameter_name real [min_value, max_value] [default value] log
    # https://automl.github.io/SMAC3/stable/options.html
    '''
    search_space = nni_search_space_content
    with open('param_config_space.pcs', 'w') as pcs_fd:
        if isinstance(search_space, dict):
            for key in search_space.keys():
                if isinstance(search_space[key], dict):
                    try:
                        if search_space[key]['_type'] == 'choice':
                            pcs_fd.write('%s categorical {%s} [%s]\n' % (
                                key, 
                                json.dumps(search_space[key]['_value'])[1:-1], 
                                json.dumps(search_space[key]['_value'][0])))
                        elif search_space[key]['_type'] == 'randint':
                            # TODO: support lower bound in randint
                            pcs_fd.write('%s integer [0, %d] [%d]\n' % (
                                key, 
                                search_space[key]['_value'][0], 
                                search_space[key]['_value'][0]))
                        elif search_space[key]['_type'] == 'uniform':
                            pcs_fd.write('%s real %s [%s]\n' % (
                                key, 
                                json.dumps(search_space[key]['_value']),
                                json.dumps(search_space[key]['_value'][0])))
                        elif search_space[key]['_type'] == 'loguniform':
                            # use np.round here to ensure that the rounded defaut value is in the range, which will be rounded in configure_space package
                            search_space[key]['_value'] = list(np.round(np.log(search_space[key]['_value']), 10))
                            pcs_fd.write('%s real %s [%s]\n' % (
                                key, 
                                json.dumps(search_space[key]['_value']),
                                json.dumps(search_space[key]['_value'][0])))
                        elif search_space[key]['_type'] == 'quniform' \
                            and search_space[key]['_value'][2] == 1:
                            pcs_fd.write('%s integer [%d, %d] [%d]\n' % (
                                key,
                                search_space[key]['_value'][0],
                                search_space[key]['_value'][1],
                                search_space[key]['_value'][0]))
                        else:
                            raise RuntimeError('unsupported _type %s' % search_space[key]['_type'])
                    except:
                        raise RuntimeError('_type or _value error.')
        else:
            raise RuntimeError('incorrect search space.')

def generate_scenario(ss_content):
    '''
    # deterministic, 1/0
    # output_dir,
    # paramfile, 
    # run_obj, 'quality'

    # the following keys use default value or empty
    # algo, not required by tuner, but required by nni's training service for running trials
    # abort_on_first_run_crash, because trials reported to nni tuner would always in success state
    # always_race_default,
    # cost_for_crash, trials reported to nni tuner would always in success state
    # cutoff_time, 
    # execdir, trials are executed by nni's training service
    # feature_file, no features specified or feature file is not supported
    # initial_incumbent, use default value
    # input_psmac_dirs, parallelism is supported by nni
    # instance_file, not supported
    # intensification_percentage, not supported, trials are controlled by nni's training service and kill be assessor
    # maxR, use default, 2000
    # minR, use default, 1
    # overall_obj, timeout is not supported
    # shared_model, parallelism is supported by nni
    # test_instance_file, instance is not supported
    # tuner-timeout, not supported
    # runcount_limit, default: inf., use default because this is controlled by nni
    # wallclock_limit,default: inf., use default because this is controlled by nni
    # please refer to https://automl.github.io/SMAC3/stable/options.html
    '''
    with open('scenario.txt', 'w') as sce_fd:
        sce_fd.write('deterministic = 0\n')
        #sce_fd.write('output_dir = \n')
        sce_fd.write('paramfile = param_config_space.pcs\n')
        sce_fd.write('run_obj = quality\n')

    generate_pcs(ss_content)

if __name__ == '__main__':
    generate_scenario('search_space.json')
