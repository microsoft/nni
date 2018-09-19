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


import json
import os
from .rest_utils import rest_put, rest_get, check_rest_server_quick, check_response
from .url_utils import experiment_url
from .config_utils import Config
from .common_utils import get_json_content

def validate_digit(value, start, end):
    '''validate if a digit is valid'''
    if not str(value).isdigit() or int(value) < start or int(value) > end:
        raise ValueError('%s must be a digit from %s to %s' % (value, start, end))

def validate_file(path):
    '''validate if a file exist'''
    if not os.path.exists(path):
        raise FileNotFoundError('%s is not a valid file path' % path)

def load_search_space(path):
    '''load search space content'''
    content = json.dumps(get_json_content(path))
    if not content:
        raise ValueError('searchSpace file should not be empty')
    return content

def get_query_type(key):
    '''get update query type'''
    if key == 'trialConcurrency':
        return '?update_type=TRIAL_CONCURRENCY'
    if key == 'maxExecDuration':
        return '?update_type=MAX_EXEC_DURATION'
    if key == 'searchSpace':
        return '?update_type=SEARCH_SPACE'

def update_experiment_profile(key, value):
    '''call restful server to update experiment profile'''
    nni_config = Config()
    rest_port = nni_config.get_config('restServerPort')
    running, _ = check_rest_server_quick(rest_port)
    if running:
        response = rest_get(experiment_url(rest_port), 20)
        if response and check_response(response):
            experiment_profile = json.loads(response.text)
            experiment_profile['params'][key] = value
            response = rest_put(experiment_url(rest_port)+get_query_type(key), json.dumps(experiment_profile), 20)
            if response and check_response(response):
                return response
    else:
        print('ERROR: restful server is not running...')
    return None

def update_searchspace(args):
    validate_file(args.filename)
    content = load_search_space(args.filename)
    if update_experiment_profile('searchSpace', content):
        print('INFO: update %s success!' % 'searchSpace')
    else:
        print('ERROR: update %s failed!' % 'searchSpace')

def update_concurrency(args):
    validate_digit(args.value, 1, 1000)
    if update_experiment_profile('trialConcurrency', int(args.value)):
        print('INFO: update %s success!' % 'concurrency')
    else:
        print('ERROR: update %s failed!' % 'concurrency')

def update_duration(args):
    validate_digit(args.value, 1, 999999999)
    if update_experiment_profile('maxExecDuration', int(args.value)):
        print('INFO: update %s success!' % 'duration')
    else:
        print('ERROR: update %s failed!' % 'duration')

