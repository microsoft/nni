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
import shutil
import string
from subprocess import Popen, PIPE, call, check_output
import tempfile
from nni_annotation import *
from .launcher_utils import validate_all_content
from .rest_utils import rest_put, rest_post, check_rest_server, check_rest_server_quick, check_response
from .url_utils import cluster_metadata_url, experiment_url, get_local_urls
from .config_utils import Config, Experiments
from .common_utils import get_yml_content, get_json_content, print_error, print_normal, print_warning, detect_process, detect_port
from .constants import *
import time
import random
import site
from pathlib import Path

def get_log_path(config_file_name):
    '''generate stdout and stderr log path'''
    stdout_full_path = os.path.join(NNICTL_HOME_DIR, config_file_name, 'stdout')
    stderr_full_path = os.path.join(NNICTL_HOME_DIR, config_file_name, 'stderr')
    return stdout_full_path, stderr_full_path

def print_log_content(config_file_name):
    '''print log information'''
    stdout_full_path, stderr_full_path = get_log_path(config_file_name)
    print_normal(' Stdout:')
    stdout_cmds = ['cat', stdout_full_path]
    stdout_content = check_output(stdout_cmds)
    print(stdout_content.decode('utf-8'))
    print('\n\n')
    print_normal(' Stderr:')
    stderr_cmds = ['cat', stderr_full_path]
    stderr_content = check_output(stderr_cmds)
    print(stderr_content.decode('utf-8'))


def start_rest_server(port, platform, mode, config_file_name, experiment_id=None):
    '''Run nni manager process'''
    nni_config = Config(config_file_name)
    if detect_port(port):
        print_error('Port %s is used by another process, please reset the port!\n' \
        'You could use \'nnictl create --help\' to get help information' % port)
        exit(1)
    
    if (platform == 'pai' or platform == 'kubeflow') and detect_port(int(port) + 1):
        print_error('PAI mode need an additional adjacent port %d, and the port %d is used by another process!\n' \
        'You could set another port to start experiment!\n' \
        'You could use \'nnictl create --help\' to get help information' % ((int(port) + 1), (int(port) + 1)))
        exit(1)

    print_normal('Starting restful server...')
    python_dir = str(Path(site.getusersitepackages()).parents[2])
    entry_file = os.path.join(python_dir, 'nni', 'main.js')
    entry_dir = os.path.join(python_dir, 'nni')
    local_entry_dir = entry_dir
    if not os.path.isfile(entry_file):
        python_dir = str(Path(site.getsitepackages()[0]).parents[2])
        entry_file = os.path.join(python_dir, 'nni', 'main.js')
        entry_dir = os.path.join(python_dir, 'nni')
        if not os.path.isfile(entry_file):
            raise Exception('Fail to find main.js under both %s and %s!' % (local_entry_dir, entry_dir))
    cmds = ['node', entry_file, '--port', str(port), '--mode', platform, '--start_mode', mode]
    if mode == 'resume':
        cmds += ['--experiment_id', experiment_id]
    stdout_full_path, stderr_full_path = get_log_path(config_file_name)
    stdout_file = open(stdout_full_path, 'a+')
    stderr_file = open(stderr_full_path, 'a+')
    time_now = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    #add time information in the header of log files
    log_header = LOG_HEADER % str(time_now)
    stdout_file.write(log_header)
    stderr_file.write(log_header)
    process = Popen(cmds, cwd=entry_dir, stdout=stdout_file, stderr=stderr_file)
    return process, str(time_now)

def set_trial_config(experiment_config, port, config_file_name):
    '''set trial configuration'''
    request_data = dict()
    request_data['trial_config'] = experiment_config['trial']
    response = rest_put(cluster_metadata_url(port), json.dumps(request_data), 20)
    if check_response(response):
        return True
    else:
        print('Error message is {}'.format(response.text))
        _, stderr_full_path = get_log_path(config_file_name)
        if response:
            with open(stderr_full_path, 'a+') as fout:
                fout.write(json.dumps(json.loads(response.text), indent=4, sort_keys=True, separators=(',', ':')))
        return False

def set_local_config(experiment_config, port, config_file_name):
    '''set local configuration'''
    return set_trial_config(experiment_config, port, config_file_name)

def set_remote_config(experiment_config, port, config_file_name):
    '''Call setClusterMetadata to pass trial'''
    #set machine_list
    request_data = dict()
    request_data['machine_list'] = experiment_config['machineList']
    response = rest_put(cluster_metadata_url(port), json.dumps(request_data), 20)
    err_message = ''
    if not response or not check_response(response):
        if response is not None:
            err_message = response.text
            _, stderr_full_path = get_log_path(config_file_name)
            with open(stderr_full_path, 'a+') as fout:
                fout.write(json.dumps(json.loads(err_message), indent=4, sort_keys=True, separators=(',', ':')))
        return False, err_message

    #set trial_config
    return set_trial_config(experiment_config, port, config_file_name), err_message

def setNNIManagerIp(experiment_config, port, config_file_name):
    '''set nniManagerIp'''
    if experiment_config.get('nniManagerIp') is None:
        return True, None
    ip_config_dict = dict()
    ip_config_dict['nni_manager_ip'] = { 'nniManagerIp' : experiment_config['nniManagerIp'] }
    response = rest_put(cluster_metadata_url(port), json.dumps(ip_config_dict), 20)
    err_message = None
    if not response or not response.status_code == 200:
        if response is not None:
            err_message = response.text
            _, stderr_full_path = get_log_path(config_file_name)
            with open(stderr_full_path, 'a+') as fout:
                fout.write(json.dumps(json.loads(err_message), indent=4, sort_keys=True, separators=(',', ':')))
        return False, err_message
    return True, None

def set_pai_config(experiment_config, port, config_file_name):
    '''set pai configuration''' 
    pai_config_data = dict()
    pai_config_data['pai_config'] = experiment_config['paiConfig']
    response = rest_put(cluster_metadata_url(port), json.dumps(pai_config_data), 20)
    err_message = None
    if not response or not response.status_code == 200:
        if response is not None:
            err_message = response.text
            _, stderr_full_path = get_log_path(config_file_name)
            with open(stderr_full_path, 'a+') as fout:
                fout.write(json.dumps(json.loads(err_message), indent=4, sort_keys=True, separators=(',', ':')))
        return False, err_message
    result, message = setNNIManagerIp(experiment_config, port, config_file_name)
    if not result:
        return result, message
    #set trial_config
    return set_trial_config(experiment_config, port, config_file_name), err_message

def set_kubeflow_config(experiment_config, port, config_file_name):
    '''set kubeflow configuration''' 
    kubeflow_config_data = dict()
    kubeflow_config_data['kubeflow_config'] = experiment_config['kubeflowConfig']
    response = rest_put(cluster_metadata_url(port), json.dumps(kubeflow_config_data), 20)
    err_message = None
    if not response or not response.status_code == 200:
        if response is not None:
            err_message = response.text
            _, stderr_full_path = get_log_path(config_file_name)
            with open(stderr_full_path, 'a+') as fout:
                fout.write(json.dumps(json.loads(err_message), indent=4, sort_keys=True, separators=(',', ':')))
        return False, err_message
    result, message = setNNIManagerIp(experiment_config, port, config_file_name)
    if not result:
        return result, message
    #set trial_config
    return set_trial_config(experiment_config, port, config_file_name), err_message

def set_experiment(experiment_config, mode, port, config_file_name):
    '''Call startExperiment (rest POST /experiment) with yaml file content'''
    request_data = dict()
    request_data['authorName'] = experiment_config['authorName']
    request_data['experimentName'] = experiment_config['experimentName']
    request_data['trialConcurrency'] = experiment_config['trialConcurrency']
    request_data['maxExecDuration'] = experiment_config['maxExecDuration']
    request_data['maxTrialNum'] = experiment_config['maxTrialNum']
    request_data['searchSpace'] = experiment_config.get('searchSpace')
    request_data['trainingServicePlatform'] = experiment_config.get('trainingServicePlatform')

    if experiment_config.get('description'):
        request_data['description'] = experiment_config['description']
    if experiment_config.get('multiPhase'):
        request_data['multiPhase'] = experiment_config.get('multiPhase')
    if experiment_config.get('multiThread'):
        request_data['multiThread'] = experiment_config.get('multiThread')
    if experiment_config.get('advisor'):
        request_data['advisor'] = experiment_config['advisor']
    else:
        request_data['tuner'] = experiment_config['tuner']
        if 'assessor' in experiment_config:
            request_data['assessor'] = experiment_config['assessor']

    request_data['clusterMetaData'] = []
    if experiment_config['trainingServicePlatform'] == 'local':
        request_data['clusterMetaData'].append(
            {'key':'codeDir', 'value':experiment_config['trial']['codeDir']})
        request_data['clusterMetaData'].append(
            {'key': 'command', 'value': experiment_config['trial']['command']})
    elif experiment_config['trainingServicePlatform'] == 'remote':
        request_data['clusterMetaData'].append(
            {'key': 'machine_list', 'value': experiment_config['machineList']})
        request_data['clusterMetaData'].append(
            {'key': 'trial_config', 'value': experiment_config['trial']})
    elif experiment_config['trainingServicePlatform'] == 'pai':
        request_data['clusterMetaData'].append(
            {'key': 'pai_config', 'value': experiment_config['paiConfig']})        
        request_data['clusterMetaData'].append(
            {'key': 'trial_config', 'value': experiment_config['trial']})
    elif experiment_config['trainingServicePlatform'] == 'kubeflow':
        request_data['clusterMetaData'].append(
            {'key': 'kubeflow_config', 'value': experiment_config['kubeflowConfig']})
        request_data['clusterMetaData'].append(
            {'key': 'trial_config', 'value': experiment_config['trial']})

    response = rest_post(experiment_url(port), json.dumps(request_data), 20)
    if check_response(response):
        return response
    else:
        _, stderr_full_path = get_log_path(config_file_name)
        if response:
            with open(stderr_full_path, 'a+') as fout:
                fout.write(json.dumps(json.loads(response.text), indent=4, sort_keys=True, separators=(',', ':')))
            print_error('Setting experiment error, error message is {}'.format(response.text))
        return None

def launch_experiment(args, experiment_config, mode, config_file_name, experiment_id=None):
    '''follow steps to start rest server and start experiment'''
    nni_config = Config(config_file_name)
    # start rest server
    rest_process, start_time = start_rest_server(args.port, experiment_config['trainingServicePlatform'], mode, config_file_name, experiment_id)
    nni_config.set_config('restServerPid', rest_process.pid)
    # Deal with annotation
    if experiment_config.get('useAnnotation'):
        path = os.path.join(tempfile.gettempdir(), 'nni', 'annotation')
        if not os.path.isdir(path):
            os.makedirs(path)
        path = tempfile.mkdtemp(dir=path)
        code_dir = expand_annotations(experiment_config['trial']['codeDir'], path)
        experiment_config['trial']['codeDir'] = code_dir
        search_space = generate_search_space(code_dir)
        experiment_config['searchSpace'] = json.dumps(search_space)
        assert search_space, ERROR_INFO % 'Generated search space is empty'
    elif experiment_config.get('searchSpacePath'):
            search_space = get_json_content(experiment_config.get('searchSpacePath'))
            experiment_config['searchSpace'] = json.dumps(search_space)
    else:
        experiment_config['searchSpace'] = json.dumps('')

    # check rest server
    running, _ = check_rest_server(args.port)
    if running:
        print_normal('Successfully started Restful server!')
    else:
        print_error('Restful server start failed!')
        print_log_content(config_file_name)
        try:
            cmds = ['kill', str(rest_process.pid)]
            call(cmds)
        except Exception:
            raise Exception(ERROR_INFO % 'Rest server stopped!')
        exit(1)

    # set remote config
    if experiment_config['trainingServicePlatform'] == 'remote':
        print_normal('Setting remote config...')
        config_result, err_msg = set_remote_config(experiment_config, args.port, config_file_name)
        if config_result:
            print_normal('Successfully set remote config!')
        else:
            print_error('Failed! Error is: {}'.format(err_msg))
            try:
                cmds = ['kill', str(rest_process.pid)]
                call(cmds)
            except Exception:
                raise Exception(ERROR_INFO % 'Rest server stopped!')
            exit(1)

    # set local config
    if experiment_config['trainingServicePlatform'] == 'local':
        print_normal('Setting local config...')
        if set_local_config(experiment_config, args.port, config_file_name):
            print_normal('Successfully set local config!')
        else:
            print_error('Set local config failed!')
            try:
                cmds = ['kill', str(rest_process.pid)]
                call(cmds)
            except Exception:
                raise Exception(ERROR_INFO % 'Rest server stopped!')
            exit(1)
    
    #set pai config
    if experiment_config['trainingServicePlatform'] == 'pai':
        print_normal('Setting pai config...')
        config_result, err_msg = set_pai_config(experiment_config, args.port, config_file_name)
        if config_result:
            print_normal('Successfully set pai config!')
        else:
            if err_msg:
                print_error('Failed! Error is: {}'.format(err_msg))
            try:
                cmds = ['kill', str(rest_process.pid)]
                call(cmds)
            except Exception:
                raise Exception(ERROR_INFO % 'Restful server stopped!')
            exit(1)
    
    #set kubeflow config
    if experiment_config['trainingServicePlatform'] == 'kubeflow':
        print_normal('Setting kubeflow config...')
        config_result, err_msg = set_kubeflow_config(experiment_config, args.port, config_file_name)
        if config_result:
            print_normal('Successfully set kubeflow config!')
        else:
            if err_msg:
                print_error('Failed! Error is: {}'.format(err_msg))
            try:
                cmds = ['pkill', '-P', str(rest_process.pid)]
                call(cmds)
            except Exception:
                raise Exception(ERROR_INFO % 'Restful server stopped!')
            exit(1)

    # start a new experiment
    print_normal('Starting experiment...')
    response = set_experiment(experiment_config, mode, args.port, config_file_name)
    if response:
        if experiment_id is None:
            experiment_id = json.loads(response.text).get('experiment_id')
        nni_config.set_config('experimentId', experiment_id)
    else:
        print_error('Start experiment failed!')
        print_log_content(config_file_name)
        try:
            cmds = ['kill', str(rest_process.pid)]
            call(cmds)
        except Exception:
            raise Exception(ERROR_INFO % 'Restful server stopped!')
        exit(1)
    if experiment_config.get('nniManagerIp'):
        web_ui_url_list = ['{0}:{1}'.format(experiment_config['nniManagerIp'], str(args.port))]
    else:
        web_ui_url_list = get_local_urls(args.port)
    nni_config.set_config('webuiUrl', web_ui_url_list)
    
    #save experiment information
    nnictl_experiment_config = Experiments()
    nnictl_experiment_config.add_experiment(experiment_id, args.port, start_time, config_file_name, experiment_config['trainingServicePlatform'])

    print_normal(EXPERIMENT_SUCCESS_INFO % (experiment_id, '   '.join(web_ui_url_list)))

def resume_experiment(args):
    '''resume an experiment'''
    experiment_config = Experiments()
    experiment_dict = experiment_config.get_all_experiments()
    experiment_id = None
    experiment_endTime = None
    #find the latest stopped experiment
    if not args.id:
        print_error('Please set experiment id! \nYou could use \'nnictl resume {id}\' to resume a stopped experiment!\n' \
        'You could use \'nnictl experiment list all\' to show all of stopped experiments!')
        exit(1)
    else:
        if experiment_dict.get(args.id) is None:
            print_error('Id %s not exist!' % args.id)
            exit(1)
        if experiment_dict[args.id]['status'] == 'running':
            print_error('Experiment %s is running!' % args.id)
            exit(1)
        experiment_id = args.id
    print_normal('Resuming experiment %s...' % experiment_id)
    nni_config = Config(experiment_dict[experiment_id]['fileName'])
    experiment_config = nni_config.get_config('experimentConfig')
    experiment_id = nni_config.get_config('experimentId')
    new_config_file_name = ''.join(random.sample(string.ascii_letters + string.digits, 8))
    new_nni_config = Config(new_config_file_name)
    new_nni_config.set_config('experimentConfig', experiment_config)
    launch_experiment(args, experiment_config, 'resume', new_config_file_name, experiment_id)
    new_nni_config.set_config('restServerPort', args.port)

def create_experiment(args):
    '''start a new experiment'''
    config_file_name = ''.join(random.sample(string.ascii_letters + string.digits, 8))
    nni_config = Config(config_file_name)
    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        print_error('Please set correct config path!')
        exit(1)
    experiment_config = get_yml_content(config_path)
    validate_all_content(experiment_config, config_path)

    nni_config.set_config('experimentConfig', experiment_config)
    launch_experiment(args, experiment_config, 'new', config_file_name)
    nni_config.set_config('restServerPort', args.port)
