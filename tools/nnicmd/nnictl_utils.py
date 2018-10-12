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
import psutil
import json
import datetime
from subprocess import call, check_output
from .rest_utils import rest_get, rest_delete, check_rest_server_quick, check_response
from .config_utils import Config, Experiments
from .url_utils import trial_jobs_url, experiment_url, trial_job_id_url
from .constants import HOME_DIR
import time
from .common_utils import print_normal, print_error, detect_process

def get_experiment_port(args):
    experiment_config = Experiments()
    experiment_dict = experiment_config.get_all_experiments()

    if not experiment_dict:
        print_normal('Experiment is not running...')
        return None
    if not args.id and len(experiment_dict.keys()) > 1:
        print_normal('There are multiple experiments running, please set the experiment id...\n')
        print('Experiment id: ', '   '.join(list(str(id) for id in experiment_dict.keys())))
        return None
    if not args.id:
        return list(experiment_dict.values())[0]
    if experiment_dict.get(args.id):
        return experiment_dict[args.id]
    else:
        print_error('Id not correct!')     
        return None

def get_experiment_id(args):
    experiment_config = Experiments()
    experiment_dict = experiment_config.get_all_experiments()
    if not args.id:
        return list(experiment_dict.keys())[0]
    return args.id

def convert_time_stamp_to_date(content):
    '''Convert time stamp to date time format'''
    start_time_stamp = content.get('startTime')
    end_time_stamp = content.get('endTime')
    if start_time_stamp:
        start_time = datetime.datetime.utcfromtimestamp(start_time_stamp // 1000).strftime("%Y/%m/%d %H:%M:%S")
        content['startTime'] = str(start_time)
    if end_time_stamp:
        end_time = datetime.datetime.utcfromtimestamp(end_time_stamp // 1000).strftime("%Y/%m/%d %H:%M:%S")
        content['endTime'] = str(end_time)
    return content

def check_rest(args):
    '''check if restful server is running'''
    port = get_experiment_port(args)
    if port is None:
        return None
    nni_config = Config(port)
    rest_port = nni_config.get_config('restServerPort')
    running, _ = check_rest_server_quick(rest_port)
    if not running:
        print_normal('Restful server is running...')
    else:
        print_normal('Restful server is not running...')

def stop_experiment(args):
    '''Stop the experiment which is running'''
    port = get_experiment_port(args)
    if port is None:
        return None
    print_normal('Stoping experiment...')
    nni_config = Config(port)
    rest_port = nni_config.get_config('restServerPort')
    rest_pid = nni_config.get_config('restServerPid')
    if not detect_process(rest_pid):
        print_normal('Experiment is not running...')
        return
    running, _ = check_rest_server_quick(rest_port)
    stop_rest_result = True
    if running:
        response = rest_delete(experiment_url(rest_port), 20)
        if not response or not check_response(response):
            print_error('Stop experiment failed!')
            stop_rest_result = False
    #sleep to wait rest handler done
    time.sleep(3)
    rest_pid = nni_config.get_config('restServerPid')
    if rest_pid:
        cmds = ['pkill', '-P', str(rest_pid)]
        call(cmds)
    if stop_rest_result:
        print_normal('Stop experiment success!')
    experiment_config = Experiments()
    experiment_config.remove_experiment(get_experiment_id(args))

def trial_ls(args):
    '''List trial'''
    port = get_experiment_port(args)
    if port is None:
        return None
    nni_config = Config(port)
    rest_port = nni_config.get_config('restServerPort')
    rest_pid = nni_config.get_config('restServerPid')
    if not detect_process(rest_pid):
        print_error('Experiment is not running...')
        return
    running, response = check_rest_server_quick(rest_port)
    if running:
        response = rest_get(trial_jobs_url(rest_port), 20)
        if response and check_response(response):
            content = json.loads(response.text)
            for index, value in enumerate(content):               
                content[index] = convert_time_stamp_to_date(value)
            print(json.dumps(content, indent=4, sort_keys=True, separators=(',', ':')))
        else:
            print_error('List trial failed...')
    else:
        print_error('Restful server is not running...')

def trial_kill(args):
    '''List trial'''
    port = get_experiment_port(args)
    if port is None:
        return None
    nni_config = Config(port)
    rest_port = nni_config.get_config('restServerPort')
    rest_pid = nni_config.get_config('restServerPid')
    if not detect_process(rest_pid):
        print_error('Experiment is not running...')
        return
    running, _ = check_rest_server_quick(rest_port)
    if running:
        response = rest_delete(trial_job_id_url(rest_port, args.trialid), 20)
        if response and check_response(response):
            print(response.text)
        else:
            print_error('Kill trial job failed...')
    else:
        print_error('Restful server is not running...')

def list_experiment(args):
    '''Get experiment information'''
    port = get_experiment_port(args)
    if port is None:
        return None
    nni_config = Config(port)
    rest_port = nni_config.get_config('restServerPort')
    rest_pid = nni_config.get_config('restServerPid')
    if not detect_process(rest_pid):
        print_error('Experiment is not running...')
        return
    running, _ = check_rest_server_quick(rest_port)
    if running:
        response = rest_get(experiment_url(rest_port), 20)
        if response and check_response(response):
            content = convert_time_stamp_to_date(json.loads(response.text))
            print(json.dumps(content, indent=4, sort_keys=True, separators=(',', ':')))
        else:
            print_error('List experiment failed...')
    else:
        print_error('Restful server is not running...')

def experiment_status(args):
    '''Show the status of experiment'''
    port = get_experiment_port(args)
    if port is None:
        return None
    nni_config = Config(port)
    rest_port = nni_config.get_config('restServerPort')
    result, response = check_rest_server_quick(rest_port)
    if not result:
        print_normal('Restful server is not running...')
    else:
        print(json.dumps(json.loads(response.text), indent=4, sort_keys=True, separators=(',', ':')))

def get_log_content(file_name, cmds):
    '''use cmds to read config content'''
    if os.path.exists(file_name):
        rest = check_output(cmds)
        print(rest.decode('utf-8'))
    else:
        print_normal('NULL!')

def log_internal(args, filetype):
    '''internal function to call get_log_content'''
    port = get_experiment_port(args)
    if port is None:
        return None
    if filetype == 'stdout':
        file_full_path = os.path.join(HOME_DIR, str(port), 'stdout')
    else:
        file_full_path = os.path.join(HOME_DIR, str(port), 'stderr')
    if args.head:
        get_log_content(file_full_path, ['head', '-' + str(args.head), file_full_path])
    elif args.tail:
        get_log_content(file_full_path, ['tail', '-' + str(args.tail), file_full_path])
    elif args.path:
        print_normal('The path of stdout file is: ' + file_full_path)
    else:
        get_log_content(file_full_path, ['cat', file_full_path])

def log_stdout(args):
    '''get stdout log'''
    log_internal(args, 'stdout')

def log_stderr(args):
    '''get stderr log'''
    log_internal(args, 'stderr')

def log_trial(args):
    ''''get trial log path'''
    trial_id_path_dict = {}
    port = get_experiment_port(args)
    if port is None:
        return None
    nni_config = Config(port)
    rest_port = nni_config.get_config('restServerPort')
    rest_pid = nni_config.get_config('restServerPid')
    if not detect_process(rest_pid):
        print_error('Experiment is not running...')
        return
    running, response = check_rest_server_quick(rest_port)
    if running:
        response = rest_get(trial_jobs_url(rest_port), 20)
        if response and check_response(response):
            content = json.loads(response.text)
            for trial in content:
                trial_id_path_dict[trial['id']] = trial['logPath']
    else:
        print_error('Restful server is not running...')
        exit(1)
    if args.id:
        if trial_id_path_dict.get(args.id):
            print('id:' + args.id + ' path:' + trial_id_path_dict[args.id])
        else:
            print_error('trial id is not valid!')
            exit(1)
    else:
        for key in trial_id_path_dict.keys():
            print('id:' + key + ' path:' + trial_id_path_dict[key])

def get_config(args):
    '''get config info'''
    nni_config = Config(args.port)
    print(nni_config.get_all_config())

def webui_url(args):
    '''show the url of web ui'''
    nni_config = Config(args.port)
    print_normal('{0} {1}'.format('Web UI url:', ' '.join(nni_config.get_config('webuiUrl'))))

def experiment_id(args):
    experiment_config = Experiments()
    experiment_dict = experiment_config.get_all_experiments()
    if not experiment_dict:
        print('There is no experiment running...')
    else:
       print('Experiment id: ', '   '.join(list(str(id) for id in experiment_dict.keys())))