# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
from pathlib import Path
import sys
import string
import random
import time
import tempfile
import re
from subprocess import Popen, check_call, CalledProcessError, PIPE, STDOUT
from nni.experiment.config import ExperimentConfig, convert
from nni.tools.annotation import expand_annotations, generate_search_space
from nni.tools.package_utils import get_builtin_module_class_name
import nni_node  # pylint: disable=import-error
from .launcher_utils import validate_all_content
from .rest_utils import rest_put, rest_post, check_rest_server, check_response
from .url_utils import cluster_metadata_url, experiment_url, get_local_urls, set_prefix_url
from .config_utils import Config, Experiments
from .common_utils import get_yml_content, get_json_content, print_error, print_normal, detect_port, get_user

from .constants import NNI_HOME_DIR, ERROR_INFO, REST_TIME_OUT, EXPERIMENT_SUCCESS_INFO, LOG_HEADER
from .command_utils import check_output_command, kill_command
from .nnictl_utils import update_experiment

k8s_training_services = ['kubeflow', 'frameworkcontroller', 'adl']

def get_log_path(experiment_id):
    '''generate stdout and stderr log path'''
    os.makedirs(os.path.join(NNI_HOME_DIR, experiment_id, 'log'), exist_ok=True)
    stdout_full_path = os.path.join(NNI_HOME_DIR, experiment_id, 'log', 'nnictl_stdout.log')
    stderr_full_path = os.path.join(NNI_HOME_DIR, experiment_id, 'log', 'nnictl_stderr.log')
    return stdout_full_path, stderr_full_path

def print_log_content(config_file_name):
    '''print log information'''
    stdout_full_path, stderr_full_path = get_log_path(config_file_name)
    print_normal(' Stdout:')
    print(check_output_command(stdout_full_path))
    print('\n\n')
    print_normal(' Stderr:')
    print(check_output_command(stderr_full_path))

def start_rest_server(port, platform, mode, experiment_id, foreground=False, log_dir=None, log_level=None, url_prefix=None):
    '''Run nni manager process'''
    if detect_port(port):
        print_error('Port %s is used by another process, please reset the port!\n' \
        'You could use \'nnictl create --help\' to get help information' % port)
        exit(1)

    if (platform not in ['local', 'aml']) and detect_port(int(port) + 1):
        print_error('%s mode need an additional adjacent port %d, and the port %d is used by another process!\n' \
        'You could set another port to start experiment!\n' \
        'You could use \'nnictl create --help\' to get help information' % (platform, (int(port) + 1), (int(port) + 1)))
        exit(1)

    print_normal('Starting restful server...')

    entry_dir = nni_node.__path__[0]
    if (not entry_dir) or (not os.path.exists(entry_dir)):
        print_error('Fail to find nni under python library')
        exit(1)
    entry_file = os.path.join(entry_dir, 'main.js')

    if sys.platform == 'win32':
        node_command = os.path.join(entry_dir, 'node.exe')
    else:
        node_command = os.path.join(entry_dir, 'node')
    cmds = [node_command, '--max-old-space-size=4096', entry_file, '--port', str(port), '--mode', platform, \
            '--experiment_id', experiment_id]
    if mode == 'view':
        cmds += ['--start_mode', 'resume']
        cmds += ['--readonly', 'true']
    else:
        cmds += ['--start_mode', mode]
    if log_dir is not None:
        cmds += ['--log_dir', log_dir]
    if log_level is not None:
        cmds += ['--log_level', log_level]
    if foreground:
        cmds += ['--foreground', 'true']
    if url_prefix:
        _validate_prefix_path(url_prefix)
        set_prefix_url(url_prefix)
        cmds += ['--url_prefix', url_prefix]

    stdout_full_path, stderr_full_path = get_log_path(experiment_id)
    with open(stdout_full_path, 'a+') as stdout_file, open(stderr_full_path, 'a+') as stderr_file:
        start_time = time.time()
        time_now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
        #add time information in the header of log files
        log_header = LOG_HEADER % str(time_now)
        stdout_file.write(log_header)
        stderr_file.write(log_header)
        if sys.platform == 'win32':
            from subprocess import CREATE_NEW_PROCESS_GROUP
            if foreground:
                process = Popen(cmds, cwd=entry_dir, stdout=PIPE, stderr=STDOUT, creationflags=CREATE_NEW_PROCESS_GROUP)
            else:
                process = Popen(cmds, cwd=entry_dir, stdout=stdout_file, stderr=stderr_file, creationflags=CREATE_NEW_PROCESS_GROUP)
        else:
            if foreground:
                process = Popen(cmds, cwd=entry_dir, stdout=PIPE, stderr=PIPE)
            else:
                process = Popen(cmds, cwd=entry_dir, stdout=stdout_file, stderr=stderr_file)
    return process, int(start_time * 1000)

def set_trial_config(experiment_config, port, config_file_name):
    '''set trial configuration'''
    request_data = dict()
    request_data['trial_config'] = experiment_config['trial']
    response = rest_put(cluster_metadata_url(port), json.dumps(request_data), REST_TIME_OUT)
    if check_response(response):
        return True
    else:
        print('Error message is {}'.format(response.text))
        _, stderr_full_path = get_log_path(config_file_name)
        if response:
            with open(stderr_full_path, 'a+') as fout:
                fout.write(json.dumps(json.loads(response.text), indent=4, sort_keys=True, separators=(',', ':')))
        return False

def set_adl_config(experiment_config, port, config_file_name):
    '''set adl configuration'''
    adl_config_data = dict()
    # hack for supporting v2 config, need refactor
    adl_config_data['adl_config'] = {}
    response = rest_put(cluster_metadata_url(port), json.dumps(adl_config_data), REST_TIME_OUT)
    err_message = None
    if not response or not response.status_code == 200:
        if response is not None:
            err_message = response.text
            _, stderr_full_path = get_log_path(config_file_name)
            with open(stderr_full_path, 'a+') as fout:
                fout.write(json.dumps(json.loads(err_message), indent=4, sort_keys=True, separators=(',', ':')))
        return False, err_message
    set_V1_common_config(experiment_config, port, config_file_name)
    result, message = setNNIManagerIp(experiment_config, port, config_file_name)
    if not result:
        return result, message
    #set trial_config
    return set_trial_config(experiment_config, port, config_file_name), None

def validate_response(response, config_file_name):
    err_message = None
    if not response or not response.status_code == 200:
        if response is not None:
            err_message = response.text
            _, stderr_full_path = get_log_path(config_file_name)
            with open(stderr_full_path, 'a+') as fout:
                fout.write(json.dumps(json.loads(err_message), indent=4, sort_keys=True, separators=(',', ':')))
        print_error('Error:' + err_message)
        exit(1)

# hack to fix v1 version_check and log_collection bug, need refactor
def set_V1_common_config(experiment_config, port, config_file_name):
    version_check = True
    #debug mode should disable version check
    if experiment_config.get('debug') is not None:
        version_check = not experiment_config.get('debug')
    #validate version check
    if experiment_config.get('versionCheck') is not None:
        version_check = experiment_config.get('versionCheck')
    response = rest_put(cluster_metadata_url(port), json.dumps({'version_check': version_check}), REST_TIME_OUT)
    validate_response(response, config_file_name)
    if experiment_config.get('logCollection'):
        data = json.dumps({'log_collection': experiment_config.get('logCollection')})
        response = rest_put(cluster_metadata_url(port), data, REST_TIME_OUT)
        validate_response(response, config_file_name)

def setNNIManagerIp(experiment_config, port, config_file_name):
    '''set nniManagerIp'''
    if experiment_config.get('nniManagerIp') is None:
        return True, None
    ip_config_dict = dict()
    ip_config_dict['nni_manager_ip'] = {'nniManagerIp': experiment_config['nniManagerIp']}
    response = rest_put(cluster_metadata_url(port), json.dumps(ip_config_dict), REST_TIME_OUT)
    err_message = None
    if not response or not response.status_code == 200:
        if response is not None:
            err_message = response.text
            _, stderr_full_path = get_log_path(config_file_name)
            with open(stderr_full_path, 'a+') as fout:
                fout.write(json.dumps(json.loads(err_message), indent=4, sort_keys=True, separators=(',', ':')))
        return False, err_message
    return True, None

def set_kubeflow_config(experiment_config, port, config_file_name):
    '''set kubeflow configuration'''
    kubeflow_config_data = dict()
    kubeflow_config_data['kubeflow_config'] = experiment_config['kubeflowConfig']
    response = rest_put(cluster_metadata_url(port), json.dumps(kubeflow_config_data), REST_TIME_OUT)
    err_message = None
    if not response or not response.status_code == 200:
        if response is not None:
            err_message = response.text
            _, stderr_full_path = get_log_path(config_file_name)
            with open(stderr_full_path, 'a+') as fout:
                fout.write(json.dumps(json.loads(err_message), indent=4, sort_keys=True, separators=(',', ':')))
        return False, err_message
    set_V1_common_config(experiment_config, port, config_file_name)
    result, message = setNNIManagerIp(experiment_config, port, config_file_name)
    if not result:
        return result, message
    #set trial_config
    return set_trial_config(experiment_config, port, config_file_name), err_message

def set_frameworkcontroller_config(experiment_config, port, config_file_name):
    '''set kubeflow configuration'''
    frameworkcontroller_config_data = dict()
    frameworkcontroller_config_data['frameworkcontroller_config'] = experiment_config['frameworkcontrollerConfig']
    response = rest_put(cluster_metadata_url(port), json.dumps(frameworkcontroller_config_data), REST_TIME_OUT)
    err_message = None
    if not response or not response.status_code == 200:
        if response is not None:
            err_message = response.text
            _, stderr_full_path = get_log_path(config_file_name)
            with open(stderr_full_path, 'a+') as fout:
                fout.write(json.dumps(json.loads(err_message), indent=4, sort_keys=True, separators=(',', ':')))
        return False, err_message
    set_V1_common_config(experiment_config, port, config_file_name)
    result, message = setNNIManagerIp(experiment_config, port, config_file_name)
    if not result:
        return result, message
    #set trial_config
    return set_trial_config(experiment_config, port, config_file_name), err_message

def set_shared_storage(experiment_config, port, config_file_name):
    if 'sharedStorage' in experiment_config:
        data = json.dumps({'shared_storage_config': experiment_config['sharedStorage']})
        response = rest_put(cluster_metadata_url(port), data, REST_TIME_OUT)
        err_message = None
        if not response or not response.status_code == 200:
            if response is not None:
                err_message = response.text
                _, stderr_full_path = get_log_path(config_file_name)
                with open(stderr_full_path, 'a+') as fout:
                    fout.write(json.dumps(json.loads(err_message), indent=4, sort_keys=True, separators=(',', ':')))
            return False, err_message
    return True, None

def set_experiment_v1(experiment_config, mode, port, config_file_name):
    '''Call startExperiment (rest POST /experiment) with yaml file content'''
    request_data = dict()
    request_data['authorName'] = experiment_config['authorName']
    request_data['experimentName'] = experiment_config['experimentName']
    request_data['trialConcurrency'] = experiment_config['trialConcurrency']
    request_data['maxExecDuration'] = experiment_config['maxExecDuration']
    request_data['maxExperimentDuration'] = str(experiment_config['maxExecDuration']) + 's'
    request_data['maxTrialNum'] = experiment_config['maxTrialNum']
    request_data['maxTrialNumber'] = experiment_config['maxTrialNum']
    request_data['searchSpace'] = experiment_config.get('searchSpace')
    request_data['trainingServicePlatform'] = experiment_config.get('trainingServicePlatform')
    # hack for hotfix, fix config.trainingService undefined error, need refactor
    request_data['trainingService'] = {'platform': experiment_config.get('trainingServicePlatform')}
    if experiment_config.get('description'):
        request_data['description'] = experiment_config['description']
    if experiment_config.get('multiPhase'):
        request_data['multiPhase'] = experiment_config.get('multiPhase')
    if experiment_config.get('multiThread'):
        request_data['multiThread'] = experiment_config.get('multiThread')
    if experiment_config.get('nniManagerIp'):
        request_data['nniManagerIp'] = experiment_config.get('nniManagerIp')
    if experiment_config.get('advisor'):
        request_data['advisor'] = experiment_config['advisor']
        if request_data['advisor'].get('gpuNum'):
            print_error('gpuNum is deprecated, please use gpuIndices instead.')
        if request_data['advisor'].get('gpuIndices') and isinstance(request_data['advisor'].get('gpuIndices'), int):
            request_data['advisor']['gpuIndices'] = str(request_data['advisor'].get('gpuIndices'))
    else:
        request_data['tuner'] = experiment_config['tuner']
        if request_data['tuner'].get('gpuNum'):
            print_error('gpuNum is deprecated, please use gpuIndices instead.')
        if request_data['tuner'].get('gpuIndices') and isinstance(request_data['tuner'].get('gpuIndices'), int):
            request_data['tuner']['gpuIndices'] = str(request_data['tuner'].get('gpuIndices'))
        if 'assessor' in experiment_config:
            request_data['assessor'] = experiment_config['assessor']
            if request_data['assessor'].get('gpuNum'):
                print_error('gpuNum is deprecated, please remove it from your config file.')
    #debug mode should disable version check
    if experiment_config.get('debug') is not None:
        request_data['versionCheck'] = not experiment_config.get('debug')
    #validate version check
    if experiment_config.get('versionCheck') is not None:
        request_data['versionCheck'] = experiment_config.get('versionCheck')
    if experiment_config.get('logCollection'):
        request_data['logCollection'] = experiment_config.get('logCollection')
    request_data['clusterMetaData'] = []
    if experiment_config['trainingServicePlatform'] == 'kubeflow':
        request_data['clusterMetaData'].append(
            {'key': 'kubeflow_config', 'value': experiment_config['kubeflowConfig']})
        request_data['clusterMetaData'].append(
            {'key': 'trial_config', 'value': experiment_config['trial']})
    elif experiment_config['trainingServicePlatform'] == 'frameworkcontroller':
        request_data['clusterMetaData'].append(
            {'key': 'frameworkcontroller_config', 'value': experiment_config['frameworkcontrollerConfig']})
        request_data['clusterMetaData'].append(
            {'key': 'trial_config', 'value': experiment_config['trial']})
    elif experiment_config['trainingServicePlatform'] == 'adl':
        request_data['clusterMetaData'].append(
            {'key': 'trial_config', 'value': experiment_config['trial']})
    response = rest_post(experiment_url(port), json.dumps(request_data), REST_TIME_OUT, show_error=True)
    if check_response(response):
        return response
    else:
        _, stderr_full_path = get_log_path(config_file_name)
        if response is not None:
            with open(stderr_full_path, 'a+') as fout:
                fout.write(json.dumps(json.loads(response.text), indent=4, sort_keys=True, separators=(',', ':')))
            print_error('Setting experiment error, error message is {}'.format(response.text))
        return None

def set_experiment_v2(experiment_config, mode, port, config_file_name):
    '''Call startExperiment (rest POST /experiment) with yaml file content'''
    response = rest_post(experiment_url(port), json.dumps(experiment_config), REST_TIME_OUT, show_error=True)
    if check_response(response):
        return response
    else:
        _, stderr_full_path = get_log_path(config_file_name)
        if response is not None:
            with open(stderr_full_path, 'a+') as fout:
                fout.write(json.dumps(json.loads(response.text), indent=4, sort_keys=True, separators=(',', ':')))
            print_error('Setting experiment error, error message is {}'.format(response.text))
        return None

def set_platform_config(platform, experiment_config, port, config_file_name, rest_process):
    '''call set_cluster_metadata for specific platform'''
    print_normal('Setting {0} config...'.format(platform))
    config_result, err_msg = None, None
    if platform == 'adl':
        config_result, err_msg = set_adl_config(experiment_config, port, config_file_name)
    elif platform == 'kubeflow':
        config_result, err_msg = set_kubeflow_config(experiment_config, port, config_file_name)
    elif platform == 'frameworkcontroller':
        config_result, err_msg = set_frameworkcontroller_config(experiment_config, port, config_file_name)
    else:
        raise Exception(ERROR_INFO % 'Unsupported platform!')
        exit(1)
    if config_result:
        config_result, err_msg = set_shared_storage(experiment_config, port, config_file_name)
    if config_result:
        print_normal('Successfully set {0} config!'.format(platform))
    else:
        print_error('Failed! Error is: {}'.format(err_msg))
        try:
            kill_command(rest_process.pid)
        except Exception:
            raise Exception(ERROR_INFO % 'Rest server stopped!')
        exit(1)

def launch_experiment(args, experiment_config, mode, experiment_id, config_version):
    '''follow steps to start rest server and start experiment'''
    # check packages for tuner
    package_name, module_name = None, None
    if experiment_config.get('tuner') and experiment_config['tuner'].get('builtinTunerName'):
        package_name = experiment_config['tuner']['builtinTunerName']
        module_name, _ = get_builtin_module_class_name('tuners', package_name)
    elif experiment_config.get('advisor') and experiment_config['advisor'].get('builtinAdvisorName'):
        package_name = experiment_config['advisor']['builtinAdvisorName']
        module_name, _ = get_builtin_module_class_name('advisors', package_name)
    if package_name and module_name:
        try:
            stdout_full_path, stderr_full_path = get_log_path(experiment_id)
            with open(stdout_full_path, 'a+') as stdout_file, open(stderr_full_path, 'a+') as stderr_file:
                check_call([sys.executable, '-c', 'import %s'%(module_name)], stdout=stdout_file, stderr=stderr_file)
        except CalledProcessError:
            print_error('some errors happen when import package %s.' %(package_name))
            print_log_content(experiment_id)
            if package_name in ['SMAC', 'BOHB', 'PPOTuner']:
                print_error(f'The dependencies for {package_name} can be installed through pip install nni[{package_name}]')
            raise
    if config_version == 1:
        log_dir = experiment_config['logDir'] if experiment_config.get('logDir') else NNI_HOME_DIR
    else:
        log_dir = experiment_config['experimentWorkingDirectory'] if experiment_config.get('experimentWorkingDirectory') else NNI_HOME_DIR
    log_level = experiment_config['logLevel'] if experiment_config.get('logLevel') else 'info'
    #view experiment mode do not need debug function, when view an experiment, there will be no new logs created
    foreground = False
    if mode != 'view':
        foreground = args.foreground
        if log_level not in ['trace', 'debug'] and (args.debug or experiment_config.get('debug') is True):
            log_level = 'debug'
    # start rest server
    if config_version == 1:
        platform = experiment_config['trainingServicePlatform']
    elif isinstance(experiment_config['trainingService'], list):
        platform = 'hybrid'
    else:
        platform = experiment_config['trainingService']['platform']

    rest_process, start_time = start_rest_server(args.port, platform, \
                                                 mode, experiment_id, foreground, log_dir, log_level, args.url_prefix)
    # save experiment information
    Experiments().add_experiment(experiment_id, args.port, start_time,
                                 platform,
                                 experiment_config.get('experimentName', 'N/A')
                                 , pid=rest_process.pid, logDir=log_dir, prefixUrl=args.url_prefix)
    # Deal with annotation
    if experiment_config.get('useAnnotation'):
        path = os.path.join(tempfile.gettempdir(), get_user(), 'nni', 'annotation')
        if not os.path.isdir(path):
            os.makedirs(path)
        path = tempfile.mkdtemp(dir=path)
        if config_version == 1:
            nas_mode = experiment_config['trial'].get('nasMode', 'classic_mode')
            code_dir = expand_annotations(experiment_config['trial']['codeDir'], path, nas_mode=nas_mode)
            experiment_config['trial']['codeDir'] = code_dir
        else:
            code_dir = expand_annotations(experiment_config['trialCodeDirectory'], path)
            experiment_config['trialCodeDirectory'] = code_dir
        search_space = generate_search_space(code_dir)
        experiment_config['searchSpace'] = search_space
        assert search_space, ERROR_INFO % 'Generated search space is empty'
    elif config_version == 1:
        if experiment_config.get('searchSpacePath'):
            search_space = get_json_content(experiment_config.get('searchSpacePath'))
            experiment_config['searchSpace'] = search_space
        else:
            experiment_config['searchSpace'] = ''

    # check rest server
    running, _ = check_rest_server(args.port)
    if running:
        print_normal('Successfully started Restful server!')
    else:
        print_error('Restful server start failed!')
        print_log_content(experiment_id)
        try:
            kill_command(rest_process.pid)
        except Exception:
            raise Exception(ERROR_INFO % 'Rest server stopped!')
        exit(1)
    if config_version == 1 and mode != 'view':
        # set platform configuration
        set_platform_config(experiment_config['trainingServicePlatform'], experiment_config, args.port,\
                            experiment_id, rest_process)

    # start a new experiment
    print_normal('Starting experiment...')
    # set debug configuration
    if mode != 'view' and experiment_config.get('debug') is None:
        experiment_config['debug'] = args.debug
    if config_version == 1:
        response = set_experiment_v1(experiment_config, mode, args.port, experiment_id)
    else:
        response = set_experiment_v2(experiment_config, mode, args.port, experiment_id)
    if response:
        if experiment_id is None:
            experiment_id = json.loads(response.text).get('experiment_id')
    else:
        print_error('Start experiment failed!')
        print_log_content(experiment_id)
        try:
            kill_command(rest_process.pid)
        except Exception:
            raise Exception(ERROR_INFO % 'Restful server stopped!')
        exit(1)
    url_prefix_format = '' if args.url_prefix is None else '/{0}'.format(args.url_prefix)
    if experiment_config.get('nniManagerIp'):
        web_ui_url_list = ['http://{0}:{1}{2}'.format(experiment_config['nniManagerIp'], str(args.port), url_prefix_format)]
    else:
        web_ui_url_list = get_local_urls(args.port, url_prefix_format)
    Experiments().update_experiment(experiment_id, 'webuiUrl', web_ui_url_list)

    print_normal(EXPERIMENT_SUCCESS_INFO % (experiment_id, '   '.join(web_ui_url_list)))
    if mode != 'view' and args.foreground:
        try:
            while True:
                log_content = rest_process.stdout.readline().strip().decode('utf-8')
                print(log_content)
        except KeyboardInterrupt:
            kill_command(rest_process.pid)
            print_normal('Stopping experiment...')

def _validate_v1(config, path):
    try:
        validate_all_content(config, path)
    except Exception as e:
        print_error(f'Config V1 validation failed: {repr(e)}')
        exit(1)

def _validate_v2(config, path):
    base_path = Path(path).parent
    try:
        conf = ExperimentConfig(_base_path=base_path, **config)
        return conf.json()
    except Exception as e:
        print_error(f'Config V2 validation failed: {repr(e)}')

def _validate_prefix_path(path):
    assert not path.startswith('/'), 'URL prefix should not start with "/".'
    parts = path.split('/')
    valid = all(re.match('^[A-Za-z0-9_-]*$', part) for part in parts)
    assert valid, 'URL prefix should only contain letter, number, underscore, and hyphen.'

def create_experiment(args):
    '''start a new experiment'''
    experiment_id = ''.join(random.sample(string.ascii_letters + string.digits, 8))
    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        print_error('Please set correct config path!')
        exit(1)
    config_yml = get_yml_content(config_path)

    if 'trainingServicePlatform' in config_yml:
        _validate_v1(config_yml, config_path)
        platform = config_yml['trainingServicePlatform']
        if platform in k8s_training_services:
            schema = 1
            config_v1 = config_yml
        else:
            schema = 2
            config_v2 = convert.to_v2(config_yml).json()
    else:
        config_v2 = _validate_v2(config_yml, config_path)
        schema = 2

    try:
        if schema == 1:
            launch_experiment(args, config_v1, 'new', experiment_id, 1)
        else:
            launch_experiment(args, config_v2, 'new', experiment_id, 2)
    except Exception as exception:
        restServerPid = Experiments().get_all_experiments().get(experiment_id, {}).get('pid')
        if restServerPid:
            kill_command(restServerPid)
        print_error(exception)
        exit(1)

def manage_stopped_experiment(args, mode):
    '''view a stopped experiment'''
    update_experiment()
    experiments_config = Experiments()
    experiments_dict = experiments_config.get_all_experiments()
    experiment_id = None
    #find the latest stopped experiment
    if not args.id:
        print_error('Please set experiment id! \nYou could use \'nnictl {0} id\' to {0} a stopped experiment!\n' \
        'You could use \'nnictl experiment list --all\' to show all experiments!'.format(mode))
        exit(1)
    else:
        if experiments_dict.get(args.id) is None:
            print_error('Id %s not exist!' % args.id)
            exit(1)
        if experiments_dict[args.id]['status'] != 'STOPPED':
            print_error('Only stopped experiments can be {0}ed!'.format(mode))
            exit(1)
        experiment_id = args.id
    print_normal('{0} experiment {1}...'.format(mode, experiment_id))
    experiment_config = Config(experiment_id, experiments_dict[args.id]['logDir']).get_config()
    experiments_config.update_experiment(args.id, 'port', args.port)
    args.url_prefix = experiments_dict[args.id]['prefixUrl']
    assert 'trainingService' in experiment_config or 'trainingServicePlatform' in experiment_config
    try:
        if 'trainingServicePlatform' in experiment_config:
            experiment_config['logDir'] = experiments_dict[args.id]['logDir']
            launch_experiment(args, experiment_config, mode, experiment_id, 1)
        else:
            experiment_config['experimentWorkingDirectory'] = experiments_dict[args.id]['logDir']
            launch_experiment(args, experiment_config, mode, experiment_id, 2)
    except Exception as exception:
        restServerPid = Experiments().get_all_experiments().get(experiment_id, {}).get('pid')
        if restServerPid:
            kill_command(restServerPid)
        print_error(exception)
        exit(1)

def view_experiment(args):
    '''view a stopped experiment'''
    manage_stopped_experiment(args, 'view')

def resume_experiment(args):
    '''resume an experiment'''
    manage_stopped_experiment(args, 'resume')
