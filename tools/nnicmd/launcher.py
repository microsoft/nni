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
from subprocess import Popen, PIPE, call
import tempfile
from nni_annotation import *
from .launcher_utils import validate_all_content
from .rest_utils import rest_put, rest_post, check_rest_server, check_rest_server_quick, check_response
from .url_utils import cluster_metadata_url, experiment_url
from .config_utils import Config
from .common_utils import get_yml_content, get_json_content, print_error, print_normal, print_warning, detect_process
from .constants import *
from .webui_utils import start_web_ui, check_web_ui

def start_rest_server(port, platform, mode, experiment_id=None):
    '''Run nni manager process'''
    print_normal('Checking environment...')
    nni_config = Config()
    rest_port = nni_config.get_config('restServerPort')
    running, _ = check_rest_server_quick(rest_port)
    if rest_port and running:
        print_error('There is an experiment running, please stop it first...')
        print_normal('You can use \'nnictl stop\' command to stop an experiment!')
        exit(0)

    print_normal('Starting restful server...')
    manager = os.environ.get('NNI_MANAGER', 'nnimanager')
    cmds = [manager, '--port', str(port), '--mode', platform, '--start_mode', mode]
    if mode == 'resume':
        cmds += ['--experiment_id', experiment_id]
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    stdout_file = open(STDOUT_FULL_PATH, 'a+')
    stderr_file = open(STDERR_FULL_PATH, 'a+')
    process = Popen(cmds, stdout=stdout_file, stderr=stderr_file)
    return process

def set_trial_config(experiment_config, port):
    '''set trial configuration'''
    request_data = dict()
    value_dict = dict()
    value_dict['command'] = experiment_config['trial']['command']
    value_dict['codeDir'] = experiment_config['trial']['codeDir']
    value_dict['gpuNum'] = experiment_config['trial']['gpuNum']
    if experiment_config['trial'].get('cpuNum'):
        value_dict['cpuNum'] = experiment_config['trial']['cpuNum']
    if experiment_config['trial'].get('memoryMB'):
        value_dict['memoryMB'] = experiment_config['trial']['memoryMB']
    if experiment_config['trial'].get('image'):
        value_dict['image'] = experiment_config['trial']['image']
    if experiment_config['trial'].get('dataDir'):
        value_dict['dataDir'] = experiment_config['trial']['dataDir']
    if experiment_config['trial'].get('outputDir'):
        value_dict['outputDir'] = experiment_config['trial']['outputDir']
    request_data['trial_config'] = value_dict
    response = rest_put(cluster_metadata_url(port), json.dumps(request_data), 20)
    if check_response(response):
        return True
    else:
        print('Error message is {}'.format(response.text))
        with open(STDERR_FULL_PATH, 'a+') as fout:
            fout.write(json.dumps(json.loads(response.text), indent=4, sort_keys=True, separators=(',', ':')))
        return False

def set_local_config(experiment_config, port):
    '''set local configuration'''
    return set_trial_config(experiment_config, port)

def set_remote_config(experiment_config, port):
    '''Call setClusterMetadata to pass trial'''
    #set machine_list
    request_data = dict()
    request_data['machine_list'] = experiment_config['machineList']
    response = rest_put(cluster_metadata_url(port), json.dumps(request_data), 20)
    err_message = ''
    if not response or not check_response(response):
        if response is not None:
            err_message = response.text
            with open(STDERR_FULL_PATH, 'a+') as fout:
                fout.write(json.dumps(json.loads(err_message), indent=4, sort_keys=True, separators=(',', ':')))
        return False, err_message

    #set trial_config
    return set_trial_config(experiment_config, port), err_message

def set_pai_config(experiment_config, port):
    '''set pai configuration''' 
    pai_config_data = dict()
    pai_config_data['pai_config'] = experiment_config['paiConfig']
    response = rest_put(cluster_metadata_url(port), json.dumps(pai_config_data), 20)
    err_message = None
    if not response or not response.status_code == 200:
        if response is not None:
            err_message = response.text
            with open(STDERR_FULL_PATH, 'a+') as fout:
                fout.write(json.dumps(json.loads(err_message), indent=4, sort_keys=True, separators=(',', ':')))
        return False, err_message

    #set trial_config
    return set_trial_config(experiment_config, port), err_message

def set_experiment(experiment_config, mode, port):
    '''Call startExperiment (rest POST /experiment) with yaml file content'''
    request_data = dict()
    request_data['authorName'] = experiment_config['authorName']
    request_data['experimentName'] = experiment_config['experimentName']
    request_data['trialConcurrency'] = experiment_config['trialConcurrency']
    request_data['maxExecDuration'] = experiment_config['maxExecDuration']
    request_data['maxTrialNum'] = experiment_config['maxTrialNum']
    request_data['searchSpace'] = experiment_config.get('searchSpace')
    if experiment_config.get('multiPhase'):
        request_data['multiPhase'] = experiment_config.get('multiPhase')
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
        value_dict = dict()
        value_dict['command'] = experiment_config['trial']['command']
        value_dict['codeDir'] = experiment_config['trial']['codeDir']
        value_dict['gpuNum'] = experiment_config['trial']['gpuNum']
        request_data['clusterMetaData'].append(
            {'key': 'trial_config', 'value': value_dict})
    elif experiment_config['trainingServicePlatform'] == 'pai':
        request_data['clusterMetaData'].append(
            {'key': 'pai_config', 'value': experiment_config['paiConfig']})
        value_dict = dict()
        value_dict['command'] = experiment_config['trial']['command']
        value_dict['codeDir'] = experiment_config['trial']['codeDir']
        value_dict['gpuNum'] = experiment_config['trial']['gpuNum']
        if experiment_config['trial'].get('cpuNum'):
            value_dict['cpuNum'] = experiment_config['trial']['cpuNum']
        if experiment_config['trial'].get('memoryMB'):
            value_dict['memoryMB'] = experiment_config['trial']['memoryMB']
        if experiment_config['trial'].get('image'):
            value_dict['image'] = experiment_config['trial']['image']
        if experiment_config['trial'].get('dataDir'):
            value_dict['dataDir'] = experiment_config['trial']['dataDir']
        if experiment_config['trial'].get('outputDir'):
            value_dict['outputDir'] = experiment_config['trial']['outputDir']
        request_data['clusterMetaData'].append(
            {'key': 'trial_config', 'value': value_dict})

    response = rest_post(experiment_url(port), json.dumps(request_data), 20)
    if check_response(response):
        return response
    else:
        with open(STDERR_FULL_PATH, 'a+') as fout:
            fout.write(json.dumps(json.loads(response.text), indent=4, sort_keys=True, separators=(',', ':')))
        print_error('Setting experiment error, error message is {}'.format(response.text))
        return None

def launch_experiment(args, experiment_config, mode, webuiport, experiment_id=None):
    '''follow steps to start rest server and start experiment'''
    nni_config = Config()
    #Check if there is an experiment running
    origin_rest_pid = nni_config.get_config('restServerPid')
    if origin_rest_pid and detect_process(origin_rest_pid):
        print_error('There is an experiment running, please stop it first...')
        print_normal('You can use \'nnictl stop\' command to stop an experiment!')
        exit(0)
    # start rest server
    rest_process = start_rest_server(REST_PORT, experiment_config['trainingServicePlatform'], mode, experiment_id)
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
    running, _ = check_rest_server(REST_PORT)
    if running:
        print_normal('Successfully started Restful server!')
    else:
        print_error('Restful server start failed!')
        try:
            cmds = ['pkill', '-P', str(rest_process.pid)]
            call(cmds)
        except Exception:
            raise Exception(ERROR_INFO % 'Rest server stopped!')
        exit(0)

    # set remote config
    if experiment_config['trainingServicePlatform'] == 'remote':
        print_normal('Setting remote config...')
        config_result, err_msg = set_remote_config(experiment_config, REST_PORT)
        if config_result:
            print_normal('Success!')
        else:
            print_error('Failed! Error is: {}'.format(err_msg))
            try:
                cmds = ['pkill', '-P', str(rest_process.pid)]
                call(cmds)
            except Exception:
                raise Exception(ERROR_INFO % 'Rest server stopped!')
            exit(0)

    # set local config
    if experiment_config['trainingServicePlatform'] == 'local':
        print_normal('Setting local config...')
        if set_local_config(experiment_config, REST_PORT):
            print_normal('Successfully set local config!')
        else:
            print_error('Failed!')
            try:
                cmds = ['pkill', '-P', str(rest_process.pid)]
                call(cmds)
            except Exception:
                raise Exception(ERROR_INFO % 'Rest server stopped!')
            exit(0)
    
    #set pai config
    if experiment_config['trainingServicePlatform'] == 'pai':
        print_normal('Setting pai config...')
        config_result, err_msg = set_pai_config(experiment_config, REST_PORT)
        if config_result:
            print_normal('Successfully set pai config!')
        else:
            if err_msg:
                print_error('Failed! Error is: {}'.format(err_msg))
            try:
                cmds = ['pkill', '-P', str(rest_process.pid)]
                call(cmds)
            except Exception:
                raise Exception(ERROR_INFO % 'Restful server stopped!')
            exit(0)
    
    #start webui
    if check_web_ui():
        print_warning('{0} {1}'.format(' '.join(nni_config.get_config('webuiUrl')),'is being used, please stop it first!'))
        print_normal('You can use \'nnictl webui stop\' to stop old Web UI process...')
    else:
        print_normal('Starting Web UI...')
        webui_process = start_web_ui(webuiport)
        if webui_process:
            nni_config.set_config('webuiPid', webui_process.pid)
            print_normal('Successfully started Web UI!')

    # start a new experiment
    print_normal('Starting experiment...')
    response = set_experiment(experiment_config, mode, REST_PORT)
    if response:
        if experiment_id is None:
            experiment_id = json.loads(response.text).get('experiment_id')
        nni_config.set_config('experimentId', experiment_id)
    else:
        print_error('Failed!')
        try:
            cmds = ['pkill', '-P', str(rest_process.pid)]
            call(cmds)
            cmds = ['pkill', '-P', str(webui_process.pid)]
            call(cmds)
        except Exception:
            raise Exception(ERROR_INFO % 'Restful server stopped!')
        exit(0)
    print_normal(EXPERIMENT_SUCCESS_INFO % (experiment_id, REST_PORT, '   '.join(nni_config.get_config('webuiUrl'))))

def resume_experiment(args):
    '''resume an experiment'''
    nni_config = Config()
    experiment_config = nni_config.get_config('experimentConfig')
    experiment_id = nni_config.get_config('experimentId')
    launch_experiment(args, experiment_config, 'resume', args.webuiport, experiment_id)

def create_experiment(args):
    '''start a new experiment'''
    nni_config = Config()
    config_path = os.path.abspath(args.config)
    experiment_config = get_yml_content(config_path)
    validate_all_content(experiment_config, config_path)

    nni_config.set_config('experimentConfig', experiment_config)
    launch_experiment(args, experiment_config, 'new', args.webuiport)
    nni_config.set_config('restServerPort', REST_PORT)
