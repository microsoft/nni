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
from .rest_utils import rest_put, rest_post, rest_get, check_rest_server_quick, check_response
from .url_utils import experiment_url, import_data_url
from .config_utils import Config
from .common_utils import get_json_content, print_normal, print_error, print_warning
from .nnictl_utils import check_experiment_id, get_experiment_port, get_config_filename
from .launcher_utils import parse_time
from .constants import REST_TIME_OUT, TUNERS_SUPPORTING_IMPORT_DATA, TUNERS_NO_NEED_TO_IMPORT_DATA

def validate_digit(value, start, end):
    '''validate if a digit is valid'''
    if not str(value).isdigit() or int(value) < start or int(value) > end:
        raise ValueError('%s must be a digit from %s to %s' % (value, start, end))

def validate_file(path):
    '''validate if a file exist'''
    if not os.path.exists(path):
        raise FileNotFoundError('%s is not a valid file path' % path)

def validate_dispatcher(args):
    '''validate if the dispatcher of the experiment supports importing data'''
    nni_config = Config(get_config_filename(args)).get_config('experimentConfig')
    if nni_config.get('tuner') and nni_config['tuner'].get('builtinTunerName'):
        dispatcher_name = nni_config['tuner']['builtinTunerName']
    elif nni_config.get('advisor') and nni_config['advisor'].get('builtinAdvisorName'):
        dispatcher_name = nni_config['advisor']['builtinAdvisorName']
    else: # otherwise it should be a customized one
        return
    if dispatcher_name not in TUNERS_SUPPORTING_IMPORT_DATA:
        if dispatcher_name in TUNERS_NO_NEED_TO_IMPORT_DATA:
            print_warning("There is no need to import data for %s" % dispatcher_name)
            exit(0)
        else:
            print_error("%s does not support importing addtional data" % dispatcher_name)
            exit(1)

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
    if key == 'maxTrialNum':
        return '?update_type=MAX_TRIAL_NUM'

def update_experiment_profile(args, key, value):
    '''call restful server to update experiment profile'''
    nni_config = Config(get_config_filename(args))
    rest_port = nni_config.get_config('restServerPort')
    running, _ = check_rest_server_quick(rest_port)
    if running:
        response = rest_get(experiment_url(rest_port), REST_TIME_OUT)
        if response and check_response(response):
            experiment_profile = json.loads(response.text)
            experiment_profile['params'][key] = value
            response = rest_put(experiment_url(rest_port)+get_query_type(key), json.dumps(experiment_profile), REST_TIME_OUT)
            if response and check_response(response):
                return response
    else:
        print_error('Restful server is not running...')
    return None

def update_searchspace(args):
    validate_file(args.filename)
    content = load_search_space(args.filename)
    args.port = get_experiment_port(args)
    if args.port is not None:
        if update_experiment_profile(args, 'searchSpace', content):
            print_normal('Update %s success!' % 'searchSpace')
        else:
            print_error('Update %s failed!' % 'searchSpace')


def update_concurrency(args):
    validate_digit(args.value, 1, 1000)
    args.port = get_experiment_port(args)
    if args.port is not None:
        if update_experiment_profile(args, 'trialConcurrency', int(args.value)):
            print_normal('Update %s success!' % 'concurrency')
        else:
            print_error('Update %s failed!' % 'concurrency')

def update_duration(args):
    #parse time, change time unit to seconds 
    args.value = parse_time(args.value)
    args.port = get_experiment_port(args)
    if args.port is not None:
        if update_experiment_profile(args, 'maxExecDuration', int(args.value)):
            print_normal('Update %s success!' % 'duration')
        else:
            print_error('Update %s failed!' % 'duration')

def update_trialnum(args):
    validate_digit(args.value, 1, 999999999)
    if update_experiment_profile(args, 'maxTrialNum', int(args.value)):
        print_normal('Update %s success!' % 'trialnum')
    else:
        print_error('Update %s failed!' % 'trialnum')

def import_data(args):
    '''import additional data to the experiment'''
    validate_file(args.filename)
    validate_dispatcher(args)
    content = load_search_space(args.filename)
    args.port = get_experiment_port(args)
    if args.port is not None:
        if import_data_to_restful_server(args, content):
            pass
        else:
            print_error('Import data failed!')

def import_data_to_restful_server(args, content):
    '''call restful server to import data to the experiment'''
    nni_config = Config(get_config_filename(args))
    rest_port = nni_config.get_config('restServerPort')
    running, _ = check_rest_server_quick(rest_port)
    if running:
        response = rest_post(import_data_url(rest_port), content, REST_TIME_OUT)
        if response and check_response(response):
            return response
    else:
        print_error('Restful server is not running...')
    return None
