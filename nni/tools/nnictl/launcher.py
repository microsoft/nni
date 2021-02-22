# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import sys
import string
import random
import time
import tempfile
from subprocess import Popen, check_call, CalledProcessError, PIPE, STDOUT
from nni.experiment.config import ExperimentConfig, convert
from nni.tools.annotation import expand_annotations, generate_search_space
from nni.tools.package_utils import get_builtin_module_class_name
import nni_node  # pylint: disable=import-error
from .launcher_utils import validate_all_content
from .rest_utils import rest_put, rest_post, check_rest_server, check_response
from .url_utils import cluster_metadata_url, experiment_url, get_local_urls
from .config_utils import Config, Experiments
from .common_utils import get_yml_content, get_json_content, print_error, print_normal, print_warning, \
                          detect_port, get_user

from .constants import NNI_HOME_DIR, ERROR_INFO, REST_TIME_OUT, EXPERIMENT_SUCCESS_INFO, LOG_HEADER
from .command_utils import check_output_command, kill_command
from .nnictl_utils import update_experiment

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

def start_rest_server(port, platform, mode, experiment_id, foreground=False, log_dir=None, log_level=None):
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

#def set_dlts_config(experiment_config, port, config_file_name):
#    '''set dlts configuration'''
#    dlts_config_data = dict()
#    dlts_config_data['dlts_config'] = experiment_config['dltsConfig']
#    response = rest_put(cluster_metadata_url(port), json.dumps(dlts_config_data), REST_TIME_OUT)
#    err_message = None
#    if not response or not response.status_code == 200:
#        if response is not None:
#            err_message = response.text
#            _, stderr_full_path = get_log_path(config_file_name)
#            with open(stderr_full_path, 'a+') as fout:
#                fout.write(json.dumps(json.loads(err_message), indent=4, sort_keys=True, separators=(',', ':')))
#        return False, err_message
#    result, message = setNNIManagerIp(experiment_config, port, config_file_name)
#    if not result:
#        return result, message
#    #set trial_config
#    return set_trial_config(experiment_config, port, config_file_name), err_message

def set_experiment(experiment_config, mode, port, config_file_name):
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

def launch_experiment(args, experiment_config, mode, experiment_id):
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
    log_dir = experiment_config['logDir'] if experiment_config.get('logDir') else NNI_HOME_DIR
    log_level = experiment_config['logLevel'] if experiment_config.get('logLevel') else None
    #view experiment mode do not need debug function, when view an experiment, there will be no new logs created
    foreground = False
    if mode != 'view':
        foreground = args.foreground
        if log_level not in ['trace', 'debug'] and (args.debug or experiment_config.get('debug') is True):
            log_level = 'debug'
    # start rest server
    rest_process, start_time = start_rest_server(args.port, experiment_config['trainingService']['platform'], \
                                                 mode, experiment_id, foreground, log_dir, log_level)
    # save experiment information
    Experiments().add_experiment(experiment_id, args.port, start_time,
                                 experiment_config['trainingService']['platform'],
                                 experiment_config.get('experimentName', 'N/A'), pid=rest_process.pid, logDir=log_dir)
    # Deal with annotation
    if experiment_config.get('useAnnotation'):
        path = os.path.join(tempfile.gettempdir(), get_user(), 'nni', 'annotation')
        if not os.path.isdir(path):
            os.makedirs(path)
        path = tempfile.mkdtemp(dir=path)
        nas_mode = experiment_config['trial'].get('nasMode', 'classic_mode')
        code_dir = expand_annotations(experiment_config['trial']['codeDir'], path, nas_mode=nas_mode)
        experiment_config['trial']['codeDir'] = code_dir
        search_space = generate_search_space(code_dir)
        experiment_config['searchSpace'] = json.dumps(search_space)
        assert search_space, ERROR_INFO % 'Generated search space is empty'

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

    # start a new experiment
    print_normal('Starting experiment...')
    # set debug configuration
    if mode != 'view' and experiment_config.get('debug') is None:
        experiment_config['debug'] = args.debug
    response = set_experiment(experiment_config, mode, args.port, experiment_id)
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
    if experiment_config.get('nniManagerIp'):
        web_ui_url_list = ['http://{0}:{1}'.format(experiment_config['nniManagerIp'], str(args.port))]
    else:
        web_ui_url_list = get_local_urls(args.port)
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

def create_experiment(args):
    '''start a new experiment'''
    experiment_id = ''.join(random.sample(string.ascii_letters + string.digits, 8))
    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        print_error('Please set correct config path!')
        exit(1)
    config_yml = get_yml_content(config_path)

    try:
        experiment_config = ExperimentConfig(**config_yml).json()
    except Exception as error_v2:
        print_warning('Validation with V2 schema failed. Trying to convert from V1 format...')
        try:
            validate_all_content(config_yml, config_path)
        except Exception as error_v1:
            print_error(f'Convert from v1 format failed: {repr(error_v1)}')
            print_error(f'Config in v2 format validation failed: {repr(error_v2)}')
            exit(1)
        from nni.experiment.config.v1 import convert_to_v2
        experiment_config = convert_to_v2(config_yml).json()

    try:
        launch_experiment(args, experiment_config, 'new', experiment_id)
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
    try:
        launch_experiment(args, experiment_config, mode, experiment_id)
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
