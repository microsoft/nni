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
from .rest_utils import rest_get, rest_delete, check_rest_server_quick
from .config_utils import Config
from .url_utils import trial_jobs_url, experiment_url, trial_job_id_url
from .constants import STDERR_FULL_PATH, STDOUT_FULL_PATH
import time
from .common_utils import print_normal, print_error, detect_process
from .webui_utils import stop_web_ui, check_web_ui, start_web_ui

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
    nni_config = Config()
    rest_port = nni_config.get_config('restServerPort')
    if check_rest_server_quick(rest_port):
        print_normal('Restful server is running...')
    else:
        print_normal('Restful server is not running...')

def stop_experiment(args):
    '''Stop the experiment which is running'''
    print_normal('Stoping experiment...')
    nni_config = Config()
    rest_port = nni_config.get_config('restServerPort')
    rest_pid = nni_config.get_config('restServerPid')
    if not detect_process(rest_pid):
        print_normal('Experiment is not running...')
        stop_web_ui()
        return
    if check_rest_server_quick(rest_port):
        response = rest_delete(experiment_url(rest_port), 20)
        if not response or response.status_code != 200:
            print_error('Stop experiment failed!')
    #sleep to wait rest handler done
    time.sleep(3)
    rest_pid = nni_config.get_config('restServerPid')
    cmds = ['pkill', '-P', str(rest_pid)]
    call(cmds)
    stop_web_ui()
    print_normal('Stop experiment success!')

def trial_ls(args):
    '''List trial'''
    nni_config = Config()
    rest_port = nni_config.get_config('restServerPort')
    rest_pid = nni_config.get_config('restServerPid')
    if not detect_process(rest_pid):
        print_error('Experiment is not running...')
        return
    if check_rest_server_quick(rest_port):
        response = rest_get(trial_jobs_url(rest_port), 20)
        if response and response.status_code == 200:
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
    nni_config = Config()
    rest_port = nni_config.get_config('restServerPort')
    rest_pid = nni_config.get_config('restServerPid')
    if not detect_process(rest_pid):
        print_error('Experiment is not running...')
        return
    if check_rest_server_quick(rest_port):
        response = rest_delete(trial_job_id_url(rest_port, args.trialid), 20)
        if response and response.status_code == 200:
            print(response.text)
        else:
            print_error('Kill trial job failed...')
    else:
        print_error('Restful server is not running...')

def list_experiment(args):
    '''Get experiment information'''
    nni_config = Config()
    rest_port = nni_config.get_config('restServerPort')
    rest_pid = nni_config.get_config('restServerPid')
    if not detect_process(rest_pid):
        print_error('Experiment is not running...')
        return
    if check_rest_server_quick(rest_port):
        response = rest_get(experiment_url(rest_port), 20)
        if response and response.status_code == 200:
            content = convert_time_stamp_to_date(json.loads(response.text))
            print(json.dumps(content, indent=4, sort_keys=True, separators=(',', ':')))
        else:
            print_error('List experiment failed...')
    else:
        print_error('Restful server is not running...')

def get_log_content(file_name, cmds):
    '''use cmds to read config content'''
    if os.path.exists(file_name):
        rest = check_output(cmds)
        print(rest.decode('utf-8'))
    else:
        print_normal('NULL!')

def log_internal(args, filetype):
    '''internal function to call get_log_content'''
    if filetype == 'stdout':
        file_full_path = STDOUT_FULL_PATH
    else:
        file_full_path = STDERR_FULL_PATH
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

def get_config(args):
    '''get config info'''
    nni_config = Config()
    print(nni_config.get_all_config())

def start_webui(args):
    '''start web ui'''
    # start webui
    print_normal('Checking webui...')
    nni_config = Config()
    rest_pid = nni_config.get_config('restServerPid')
    if not detect_process(rest_pid):
        print_error('Experiment is not running...')
        return
    if check_web_ui():
        print_error('{0} {1}'.format(' '.join(nni_config.get_config('webuiUrl')), 'is being used, please stop it first!'))
        print_normal('You can use \'nnictl webui stop\' to stop old web ui process...')
    else:
        print_normal('Starting webui...')
        webui_process = start_web_ui(args.port)
        nni_config = Config()
        nni_config.set_config('webuiPid', webui_process.pid)
        print_normal('Starting webui success!')
        print_normal('{0} {1}'.format('Web UI url:', '   '.join(nni_config.get_config('webuiUrl'))))

def stop_webui(args):
    '''stop web ui'''
    print_normal('Stopping Web UI...')
    if stop_web_ui():
        print_normal('Web UI stopped success!')
    else:
        print_error('Web UI stop failed...')

def webui_url(args):
    '''show the url of web ui'''
    nni_config = Config()
    print_normal('{0} {1}'.format('Web UI url:', ' '.join(nni_config.get_config('webuiUrl'))))
