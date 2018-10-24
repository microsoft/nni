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
import time
from subprocess import call, check_output, Popen, PIPE
from .rest_utils import rest_get, rest_delete, check_rest_server_quick, check_response
from .config_utils import Config, Experiments
from .url_utils import trial_jobs_url, experiment_url, trial_job_id_url, get_local_urls
from .constants import NNICTL_HOME_DIR, EXPERIMENT_INFORMATION_FORMAT, EXPERIMENT_DETAIL_FORMAT
import time
from .common_utils import print_normal, print_error, print_warning, detect_process, detect_port
from .nnictl_utils import *
import re

def start_local_tensorboard_process(log_path, port, nni_config):
    '''call cmds to start tensorboard process in local mode'''
    if detect_port(port):
        print_error('Port %s is used by another process, please reset port!' % str(port))
        exit(1)
    temp_dir = os.environ['HOME']
    stdout_file = open(os.path.join(temp_dir, 'tensorboard_stdout'), 'a+')
    stderr_file = open(os.path.join(temp_dir, 'tensorboard_stderr'), 'a+')
    cmds = ['tensorboard', '--logdir', log_path, '--port', str(port)]
    tensorboard_process = Popen(cmds, stdout=stdout_file, stderr=stderr_file)
    url_list = get_local_urls(port)
    print_normal('Start tensorboard success, you can visit tensorboard from:    ' + '     '.join(url_list))
    nni_config.set_config('tensorboardPid', tensorboard_process.pid)

def stop_tensorboard(args):
    '''stop tensorboard'''
    experiment_id = check_experiment_id(args)
    experiment_config = Experiments()
    experiment_dict = experiment_config.get_all_experiments()
    config_file_name = experiment_dict[experiment_id]['fileName']
    nni_config = Config(config_file_name)
    tensorboard_pid = nni_config.get_config('tensorboardPid')
    cmds = ['kill', '-9', str(tensorboard_pid)]
    call(cmds)
    print_normal('Stop tensorboard success!')


def start_tensorboard(args):
    '''start tensorboard'''
    experiment_id = check_experiment_id(args)
    experiment_config = Experiments()
    experiment_dict = experiment_config.get_all_experiments()
    config_file_name = experiment_dict[experiment_id]['fileName']
    nni_config = Config(config_file_name)
    rest_port = nni_config.get_config('restServerPort')
    rest_pid = nni_config.get_config('restServerPid')
    if not detect_process(rest_pid):
        print_error('Experiment is not running...')
        return
    running, response = check_rest_server_quick(rest_port)
    trial_content = None
    if running:
        response = rest_get(trial_jobs_url(rest_port), 20)
        if response and check_response(response):
            trial_content = json.loads(response.text)
        else:
            print_error('List trial failed...')
    else:
        print_error('Restful server is not running...')
    if not trial_content:
        print_error('No trial information!')
        exit(1)
    path_list = []
    if args.trialid is None:
        for trial in trial_content:
            pattern = r'(?P<head>.+)://(?P<host>.+):(?P<path>.*)'
            match = re.search(pattern,trial['logPath'])
            if match:
                path_list.append(match.group('path'))
    else:
        for trial in trial_content:
            if trial.get(args.trialid):
                path_list.append(trial['logPath'])
                break
    if not path_list:
        print_error('Trial id %s error!' % args.trialid)
        exit(1)
    
    start_local_tensorboard_process(':'.join(path_list), args.port, nni_config)