# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
import re
import tempfile
from subprocess import call, Popen
from .rest_utils import rest_get, check_rest_server_quick, check_response
from .config_utils import Config, Experiments
from .url_utils import trial_jobs_url, get_local_urls
from .constants import REST_TIME_OUT
from .common_utils import print_normal, print_warning, print_error, print_green, detect_process, detect_port, check_tensorboard_version
from .nnictl_utils import check_experiment_id
from .ssh_utils import create_ssh_sftp_client, copy_remote_directory_to_local

def parse_log_path(args, trial_content):
    '''parse log path'''
    path_list = []
    host_list = []
    for trial in trial_content:
        if args.trial_id and args.trial_id != 'all' and trial.get('trialJobId') != args.trial_id:
            continue
        pattern = r'(?P<head>.+)://(?P<host>.+):(?P<path>.*)'
        match = re.search(pattern, trial['logPath'])
        if match:
            path_list.append(match.group('path'))
            host_list.append(match.group('host'))
    if not path_list:
        print_error('Trial id %s error!' % args.trial_id)
        exit(1)
    return path_list, host_list

def copy_data_from_remote(args, experiment_config, trial_content, path_list, host_list, temp_nni_path):
    '''use ssh client to copy data from remote machine to local machien'''
    machine_list = experiment_config.get('machineList')
    machine_dict = {}
    local_path_list = []
    for machine in machine_list:
        machine_dict[machine['ip']] = {'port': machine['port'], 'passwd': machine['passwd'], 'username': machine['username'],
                                       'sshKeyPath': machine.get('sshKeyPath'), 'passphrase': machine.get('passphrase')}
    for index, host in enumerate(host_list):
        local_path = os.path.join(temp_nni_path, trial_content[index].get('trialJobId'))
        local_path_list.append(local_path)
        print_normal('Copying log data from %s to %s' % (host + ':' + path_list[index], local_path))
        sftp = create_ssh_sftp_client(host, machine_dict[host]['port'], machine_dict[host]['username'], machine_dict[host]['passwd'],
                                      machine_dict[host]['sshKeyPath'], machine_dict[host]['passphrase'])
        copy_remote_directory_to_local(sftp, path_list[index], local_path)
    print_normal('Copy done!')
    return local_path_list

def get_path_list(args, experiment_config, trial_content, temp_nni_path):
    '''get path list according to different platform'''
    path_list, host_list = parse_log_path(args, trial_content)
    platform = experiment_config.get('trainingServicePlatform')
    if platform == 'local':
        print_normal('Log path: %s' % ' '.join(path_list))
        return path_list
    elif platform == 'remote':
        path_list = copy_data_from_remote(args, experiment_config, trial_content, path_list, host_list, temp_nni_path)
        print_normal('Log path: %s' % ' '.join(path_list))
        return path_list
    else:
        print_error('Not supported platform!')
        exit(1)

def format_tensorboard_log_path(path_list):
    new_path_list = []
    for index, value in enumerate(path_list):
        new_path_list.append('name%d:%s' % (index + 1, value))
    return ','.join(new_path_list)

def start_tensorboard_process(args, experiment_id, path_list, temp_nni_path):
    '''call cmds to start tensorboard process in local machine'''
    if detect_port(args.port):
        print_error('Port %s is used by another process, please reset port!' % str(args.port))
        exit(1)
    with open(os.path.join(temp_nni_path, 'tensorboard_stdout'), 'a+') as stdout_file, \
         open(os.path.join(temp_nni_path, 'tensorboard_stderr'), 'a+') as stderr_file:
        log_dir_cmd = '--logdir_spec' if check_tensorboard_version() >= '2.0' else '--logdir'
        cmds = ['tensorboard', log_dir_cmd, format_tensorboard_log_path(path_list), '--port', str(args.port)]
        tensorboard_process = Popen(cmds, stdout=stdout_file, stderr=stderr_file)
    url_list = get_local_urls(args.port)
    print_green('Start tensorboard success!')
    print_normal('Tensorboard urls: ' + '     '.join(url_list))
    experiments_config = Experiments()
    tensorboard_process_pid_list = experiments_config.get_all_experiments().get(experiment_id).get('tensorboardPidList')
    if tensorboard_process_pid_list is None:
        tensorboard_process_pid_list = [tensorboard_process.pid]
    else:
        tensorboard_process_pid_list.append(tensorboard_process.pid)
    experiments_config.update_experiment(experiment_id, 'tensorboardPidList', tensorboard_process_pid_list)

def stop_tensorboard(args):
    '''stop tensorboard'''
    experiment_id = check_experiment_id(args)
    experiments_config = Experiments()
    tensorboard_pid_list = experiments_config.get_all_experiments().get(experiment_id).get('tensorboardPidList')
    if tensorboard_pid_list:
        for tensorboard_pid in tensorboard_pid_list:
            try:
                cmds = ['kill', '-9', str(tensorboard_pid)]
                call(cmds)
            except Exception as exception:
                print_error(exception)
        experiments_config.update_experiment(experiment_id, 'tensorboardPidList', [])
        print_normal('Stop tensorboard success!')
    else:
        print_error('No tensorboard configuration!')

def adl_tensorboard_helper(args):
    '''start tensorboard on adl'''
    import subprocess
    if args.trial_id is not None:
        print_warning('Tensorboard on adl platform will show all trials. No trial ids needed.')
    cmd = "kubectl port-forward --address 0.0.0.0 deployment/{} {}:{}".format(
        "adaptdl-tensorboard" + "-" + args.id.lower(),
        args.port,
        6006
    )
    print_green('Tensorboard is accessible at 0.0.0.0:{port} or localhost:{port}'.format(port=args.port))
    subprocess.run(args=cmd, shell=True)

def start_tensorboard(args):
    '''start tensorboard'''
    experiment_id = check_experiment_id(args)
    if not experiment_id:
        return
    if args.id is None:
        args.id = experiment_id
    experiments_config = Experiments()
    experiments_dict = experiments_config.get_all_experiments()
    if experiments_dict[args.id]["status"] == "STOPPED":
        print_error("Experiment {} is stopped...".format(args.id))
        return
    experiment_config = Config(args.id, experiments_dict[args.id]['logDir']).get_config()
    if experiment_config.get('trainingServicePlatform') == 'adl':
        adl_tensorboard_helper(args)
        return
    rest_port = experiments_dict[args.id]['port']
    rest_pid = experiments_dict[args.id]['pid']
    if not detect_process(rest_pid):
        print_error('Experiment is not running...')
        return
    running, response = check_rest_server_quick(rest_port)
    trial_content = None
    if running:
        response = rest_get(trial_jobs_url(rest_port), REST_TIME_OUT)
        if response and check_response(response):
            trial_content = json.loads(response.text)
        else:
            print_error('List trial failed...')
    else:
        print_error('Restful server is not running...')
    if not trial_content:
        print_error('No trial information!')
        exit(1)
    if len(trial_content) > 1 and not args.trial_id:
        print_error('There are multiple trials, please set trial id!')
        exit(1)
    experiment_id = args.id
    temp_nni_path = os.path.join(tempfile.gettempdir(), 'nni', experiment_id)
    os.makedirs(temp_nni_path, exist_ok=True)

    path_list = get_path_list(args, experiment_config, trial_content, temp_nni_path)
    start_tensorboard_process(args, experiment_id, path_list, temp_nni_path)
