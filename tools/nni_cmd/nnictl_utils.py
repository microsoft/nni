# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import csv
import os
import sys
import json
import time
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from subprocess import Popen
from pyhdfs import HdfsClient
from nni_annotation import expand_annotations
from .rest_utils import rest_get, rest_delete, check_rest_server_quick, check_response
from .url_utils import trial_jobs_url, experiment_url, trial_job_id_url, export_data_url
from .config_utils import Config, Experiments
from .constants import NNICTL_HOME_DIR, EXPERIMENT_INFORMATION_FORMAT, EXPERIMENT_DETAIL_FORMAT, \
     EXPERIMENT_MONITOR_INFO, TRIAL_MONITOR_HEAD, TRIAL_MONITOR_CONTENT, TRIAL_MONITOR_TAIL, REST_TIME_OUT
from .common_utils import print_normal, print_error, print_warning, detect_process, get_yml_content, get_nni_installation_path
from .command_utils import check_output_command, kill_command
from .ssh_utils import create_ssh_sftp_client, remove_remote_directory

def get_experiment_time(port):
    '''get the startTime and endTime of an experiment'''
    response = rest_get(experiment_url(port), REST_TIME_OUT)
    if response and check_response(response):
        content = convert_time_stamp_to_date(json.loads(response.text))
        return content.get('startTime'), content.get('endTime')
    return None, None

def get_experiment_status(port):
    '''get the status of an experiment'''
    result, response = check_rest_server_quick(port)
    if result:
        return json.loads(response.text).get('status')
    return None

def update_experiment():
    '''Update the experiment status in config file'''
    experiment_config = Experiments()
    experiment_dict = experiment_config.get_all_experiments()
    if not experiment_dict:
        return None
    for key in experiment_dict.keys():
        if isinstance(experiment_dict[key], dict):
            if experiment_dict[key].get('status') != 'STOPPED':
                nni_config = Config(experiment_dict[key]['fileName'])
                rest_pid = nni_config.get_config('restServerPid')
                if not detect_process(rest_pid):
                    experiment_config.update_experiment(key, 'status', 'STOPPED')
                    continue
                rest_port = nni_config.get_config('restServerPort')
                startTime, endTime = get_experiment_time(rest_port)
                if startTime:
                    experiment_config.update_experiment(key, 'startTime', startTime)
                if endTime:
                    experiment_config.update_experiment(key, 'endTime', endTime)
                status = get_experiment_status(rest_port)
                if status:
                    experiment_config.update_experiment(key, 'status', status)

def check_experiment_id(args, update=True):
    '''check if the id is valid
    '''
    if update:
        update_experiment()
    experiment_config = Experiments()
    experiment_dict = experiment_config.get_all_experiments()
    if not experiment_dict:
        print_normal('There is no experiment running...')
        return None
    if not args.id:
        running_experiment_list = []
        for key in experiment_dict.keys():
            if isinstance(experiment_dict[key], dict):
                if experiment_dict[key].get('status') != 'STOPPED':
                    running_experiment_list.append(key)
            elif isinstance(experiment_dict[key], list):
                # if the config file is old version, remove the configuration from file
                experiment_config.remove_experiment(key)
        if len(running_experiment_list) > 1:
            print_error('There are multiple experiments, please set the experiment id...')
            experiment_information = ""
            for key in running_experiment_list:
                experiment_information += EXPERIMENT_DETAIL_FORMAT % (key,
                                                                      experiment_dict[key].get('experimentName', 'N/A'),
                                                                      experiment_dict[key]['status'],
                                                                      experiment_dict[key]['port'],
                                                                      experiment_dict[key].get('platform'),
                                                                      experiment_dict[key]['startTime'],
                                                                      experiment_dict[key]['endTime'])
            print(EXPERIMENT_INFORMATION_FORMAT % experiment_information)
            exit(1)
        elif not running_experiment_list:
            print_error('There is no experiment running.')
            return None
        else:
            return running_experiment_list[0]
    if experiment_dict.get(args.id):
        return args.id
    else:
        print_error('Id not correct.')
        return None

def parse_ids(args):
    '''Parse the arguments for nnictl stop
    1.If port is provided and id is not specified, return the id who owns the port
    2.If both port and id are provided, return the id if it owns the port, otherwise fail
    3.If there is an id specified, return the corresponding id
    4.If there is no id specified, and there is an experiment running, return the id, or return Error
    5.If the id matches an experiment, nnictl will return the id.
    6.If the id ends with *, nnictl will match all ids matchs the regular
    7.If the id does not exist but match the prefix of an experiment id, nnictl will return the matched id
    8.If the id does not exist but match multiple prefix of the experiment ids, nnictl will give id information
    '''
    update_experiment()
    experiment_config = Experiments()
    experiment_dict = experiment_config.get_all_experiments()
    if not experiment_dict:
        print_normal('Experiment is not running...')
        return None
    result_list = []
    running_experiment_list = []
    for key in experiment_dict.keys():
        if isinstance(experiment_dict[key], dict):
            if experiment_dict[key].get('status') != 'STOPPED':
                running_experiment_list.append(key)
        elif isinstance(experiment_dict[key], list):
            # if the config file is old version, remove the configuration from file
            experiment_config.remove_experiment(key)
    if args.all:
        return running_experiment_list
    if args.port is not None:
        for key in running_experiment_list:
            if str(experiment_dict[key]['port']) == args.port:
                result_list.append(key)
        if args.id and result_list and args.id != result_list[0]:
            print_error('Experiment id and resful server port not match')
            exit(1)
    elif not args.id:
        if len(running_experiment_list) > 1:
            print_error('There are multiple experiments, please set the experiment id...')
            experiment_information = ""
            for key in running_experiment_list:
                experiment_information += EXPERIMENT_DETAIL_FORMAT % (key,
                                                                      experiment_dict[key].get('experimentName', 'N/A'),
                                                                      experiment_dict[key]['status'],
                                                                      experiment_dict[key]['port'],
                                                                      experiment_dict[key].get('platform'),
                                                                      experiment_dict[key]['startTime'],
                                                                      experiment_dict[key]['endTime'])
            print(EXPERIMENT_INFORMATION_FORMAT % experiment_information)
            exit(1)
        else:
            result_list = running_experiment_list
    elif args.id.endswith('*'):
        for expId in running_experiment_list:
            if expId.startswith(args.id[:-1]):
                result_list.append(expId)
    elif args.id in running_experiment_list:
        result_list.append(args.id)
    else:
        for expId in running_experiment_list:
            if expId.startswith(args.id):
                result_list.append(expId)
        if len(result_list) > 1:
            print_error(args.id + ' is ambiguous, please choose ' + ' '.join(result_list))
            return None
    if not result_list and (args.id  or args.port):
        print_error('There are no experiments matched, please set correct experiment id or restful server port')
    elif not result_list:
        print_error('There is no experiment running...')
    return result_list

def get_config_filename(args):
    '''get the file name of config file'''
    experiment_id = check_experiment_id(args)
    if experiment_id is None:
        print_error('Please set correct experiment id.')
        exit(1)
    experiment_config = Experiments()
    experiment_dict = experiment_config.get_all_experiments()
    return experiment_dict[experiment_id]['fileName']

def get_experiment_port(args):
    '''get the port of experiment'''
    experiment_id = check_experiment_id(args)
    if experiment_id is None:
        print_error('Please set correct experiment id.')
        exit(1)
    experiment_config = Experiments()
    experiment_dict = experiment_config.get_all_experiments()
    return experiment_dict[experiment_id]['port']

def convert_time_stamp_to_date(content):
    '''Convert time stamp to date time format'''
    start_time_stamp = content.get('startTime')
    end_time_stamp = content.get('endTime')
    if start_time_stamp:
        start_time = datetime.fromtimestamp(start_time_stamp // 1000, timezone.utc).astimezone().strftime("%Y/%m/%d %H:%M:%S")
        content['startTime'] = str(start_time)
    if end_time_stamp:
        end_time = datetime.fromtimestamp(end_time_stamp // 1000, timezone.utc).astimezone().strftime("%Y/%m/%d %H:%M:%S")
        content['endTime'] = str(end_time)
    return content

def check_rest(args):
    '''check if restful server is running'''
    nni_config = Config(get_config_filename(args))
    rest_port = nni_config.get_config('restServerPort')
    running, _ = check_rest_server_quick(rest_port)
    if not running:
        print_normal('Restful server is running...')
    else:
        print_normal('Restful server is not running...')

def stop_experiment(args):
    '''Stop the experiment which is running'''
    if args.id and args.id == 'all':
        print_warning('\'nnictl stop all\' is abolished, please use \'nnictl stop --all\' to stop all of experiments!')
        exit(1)
    experiment_id_list = parse_ids(args)
    if experiment_id_list:
        experiment_config = Experiments()
        experiment_dict = experiment_config.get_all_experiments()
        for experiment_id in experiment_id_list:
            print_normal('Stopping experiment %s' % experiment_id)
            nni_config = Config(experiment_dict[experiment_id]['fileName'])
            rest_pid = nni_config.get_config('restServerPid')
            if rest_pid:
                kill_command(rest_pid)
                tensorboard_pid_list = nni_config.get_config('tensorboardPidList')
                if tensorboard_pid_list:
                    for tensorboard_pid in tensorboard_pid_list:
                        try:
                            kill_command(tensorboard_pid)
                        except Exception as exception:
                            print_error(exception)
                    nni_config.set_config('tensorboardPidList', [])
            print_normal('Stop experiment success.')
            experiment_config.update_experiment(experiment_id, 'status', 'STOPPED')
            time_now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            experiment_config.update_experiment(experiment_id, 'endTime', str(time_now))

def trial_ls(args):
    '''List trial'''
    nni_config = Config(get_config_filename(args))
    rest_port = nni_config.get_config('restServerPort')
    rest_pid = nni_config.get_config('restServerPid')
    if not detect_process(rest_pid):
        print_error('Experiment is not running...')
        return
    running, response = check_rest_server_quick(rest_port)
    if running:
        response = rest_get(trial_jobs_url(rest_port), REST_TIME_OUT)
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
    nni_config = Config(get_config_filename(args))
    rest_port = nni_config.get_config('restServerPort')
    rest_pid = nni_config.get_config('restServerPid')
    if not detect_process(rest_pid):
        print_error('Experiment is not running...')
        return
    running, _ = check_rest_server_quick(rest_port)
    if running:
        response = rest_delete(trial_job_id_url(rest_port, args.trial_id), REST_TIME_OUT)
        if response and check_response(response):
            print(response.text)
        else:
            print_error('Kill trial job failed...')
    else:
        print_error('Restful server is not running...')

def trial_codegen(args):
    '''Generate code for a specific trial'''
    print_warning('Currently, this command is only for nni nas programming interface.')
    exp_id = check_experiment_id(args)
    nni_config = Config(get_config_filename(args))
    if not nni_config.get_config('experimentConfig')['useAnnotation']:
        print_error('The experiment is not using annotation')
        exit(1)
    code_dir = nni_config.get_config('experimentConfig')['trial']['codeDir']
    expand_annotations(code_dir, './exp_%s_trial_%s_code'%(exp_id, args.trial_id), exp_id, args.trial_id)

def list_experiment(args):
    '''Get experiment information'''
    nni_config = Config(get_config_filename(args))
    rest_port = nni_config.get_config('restServerPort')
    rest_pid = nni_config.get_config('restServerPid')
    if not detect_process(rest_pid):
        print_error('Experiment is not running...')
        return
    running, _ = check_rest_server_quick(rest_port)
    if running:
        response = rest_get(experiment_url(rest_port), REST_TIME_OUT)
        if response and check_response(response):
            content = convert_time_stamp_to_date(json.loads(response.text))
            print(json.dumps(content, indent=4, sort_keys=True, separators=(',', ':')))
        else:
            print_error('List experiment failed...')
    else:
        print_error('Restful server is not running...')

def experiment_status(args):
    '''Show the status of experiment'''
    nni_config = Config(get_config_filename(args))
    rest_port = nni_config.get_config('restServerPort')
    result, response = check_rest_server_quick(rest_port)
    if not result:
        print_normal('Restful server is not running...')
    else:
        print(json.dumps(json.loads(response.text), indent=4, sort_keys=True, separators=(',', ':')))

def log_internal(args, filetype):
    '''internal function to call get_log_content'''
    file_name = get_config_filename(args)
    if filetype == 'stdout':
        file_full_path = os.path.join(NNICTL_HOME_DIR, file_name, 'stdout')
    else:
        file_full_path = os.path.join(NNICTL_HOME_DIR, file_name, 'stderr')
    print(check_output_command(file_full_path, head=args.head, tail=args.tail))

def log_stdout(args):
    '''get stdout log'''
    log_internal(args, 'stdout')

def log_stderr(args):
    '''get stderr log'''
    log_internal(args, 'stderr')

def log_trial(args):
    ''''get trial log path'''
    trial_id_path_dict = {}
    trial_id_list = []
    nni_config = Config(get_config_filename(args))
    rest_port = nni_config.get_config('restServerPort')
    rest_pid = nni_config.get_config('restServerPid')
    if not detect_process(rest_pid):
        print_error('Experiment is not running...')
        return
    running, response = check_rest_server_quick(rest_port)
    if running:
        response = rest_get(trial_jobs_url(rest_port), REST_TIME_OUT)
        if response and check_response(response):
            content = json.loads(response.text)
            for trial in content:
                trial_id_list.append(trial.get('id'))
                if trial.get('logPath'):
                    trial_id_path_dict[trial.get('id')] = trial['logPath']
    else:
        print_error('Restful server is not running...')
        exit(1)
    if args.trial_id:
        if args.trial_id not in trial_id_list:
            print_error('Trial id {0} not correct, please check your command!'.format(args.trial_id))
            exit(1)
        if trial_id_path_dict.get(args.trial_id):
            print_normal('id:' + args.trial_id + ' path:' + trial_id_path_dict[args.trial_id])
        else:
            print_error('Log path is not available yet, please wait...')
            exit(1)
    else:
        print_normal('All of trial log info:')
        for key in trial_id_path_dict:
            print_normal('id:' + key + ' path:' + trial_id_path_dict[key])
        if not trial_id_path_dict:
            print_normal('None')

def get_config(args):
    '''get config info'''
    nni_config = Config(get_config_filename(args))
    print(nni_config.get_all_config())

def webui_url(args):
    '''show the url of web ui'''
    nni_config = Config(get_config_filename(args))
    print_normal('{0} {1}'.format('Web UI url:', ' '.join(nni_config.get_config('webuiUrl'))))

def webui_nas(args):
    '''launch nas ui'''
    print_normal('Starting NAS UI...')
    try:
        entry_dir = get_nni_installation_path()
        entry_file = os.path.join(entry_dir, 'nasui', 'server.js')
        node_command = 'node'
        if sys.platform == 'win32':
            node_command = os.path.join(entry_dir[:-3], 'Scripts', 'node.exe')
        cmds = [node_command, '--max-old-space-size=4096', entry_file, '--port', str(args.port), '--logdir', args.logdir]
        subprocess.run(cmds)
    except KeyboardInterrupt:
        pass

def local_clean(directory):
    '''clean up local data'''
    print_normal('removing folder {0}'.format(directory))
    try:
        shutil.rmtree(directory)
    except FileNotFoundError:
        print_error('{0} does not exist.'.format(directory))

def remote_clean(machine_list, experiment_id=None):
    '''clean up remote data'''
    for machine in machine_list:
        passwd = machine.get('passwd')
        userName = machine.get('username')
        host = machine.get('ip')
        port = machine.get('port')
        sshKeyPath = machine.get('sshKeyPath')
        passphrase = machine.get('passphrase')
        if experiment_id:
            remote_dir = '/' + '/'.join(['tmp', 'nni', 'experiments', experiment_id])
        else:
            remote_dir = '/' + '/'.join(['tmp', 'nni', 'experiments'])
        sftp = create_ssh_sftp_client(host, port, userName, passwd, sshKeyPath, passphrase)
        print_normal('removing folder {0}'.format(host + ':' + str(port) + remote_dir))
        remove_remote_directory(sftp, remote_dir)

def hdfs_clean(host, user_name, output_dir, experiment_id=None):
    '''clean up hdfs data'''
    hdfs_client = HdfsClient(hosts='{0}:80'.format(host), user_name=user_name, webhdfs_path='/webhdfs/api/v1', timeout=5)
    if experiment_id:
        full_path = '/' + '/'.join([user_name, 'nni', 'experiments', experiment_id])
    else:
        full_path = '/' + '/'.join([user_name, 'nni', 'experiments'])
    print_normal('removing folder {0} in hdfs'.format(full_path))
    hdfs_client.delete(full_path, recursive=True)
    if output_dir:
        pattern = re.compile('hdfs://(?P<host>([0-9]{1,3}.){3}[0-9]{1,3})(:[0-9]{2,5})?(?P<baseDir>/.*)?')
        match_result = pattern.match(output_dir)
        if match_result:
            output_host = match_result.group('host')
            output_dir = match_result.group('baseDir')
            #check if the host is valid
            if output_host != host:
                print_warning('The host in {0} is not consistent with {1}'.format(output_dir, host))
            else:
                if experiment_id:
                    output_dir = output_dir + '/' + experiment_id
                print_normal('removing folder {0} in hdfs'.format(output_dir))
                hdfs_client.delete(output_dir, recursive=True)

def experiment_clean(args):
    '''clean up the experiment data'''
    experiment_id_list = []
    experiment_config = Experiments()
    experiment_dict = experiment_config.get_all_experiments()
    if args.all:
        experiment_id_list = list(experiment_dict.keys())
    else:
        if args.id is None:
            print_error('please set experiment id.')
            exit(1)
        if args.id not in experiment_dict:
            print_error('Cannot find experiment {0}.'.format(args.id))
            exit(1)
        experiment_id_list.append(args.id)
    while True:
        print('INFO: This action will delete experiment {0}, and it\'s not recoverable.'.format(' '.join(experiment_id_list)))
        inputs = input('INFO: do you want to continue?[y/N]:')
        if not inputs.lower() or inputs.lower() in ['n', 'no']:
            exit(0)
        elif inputs.lower() not in ['y', 'n', 'yes', 'no']:
            print_warning('please input Y or N.')
        else:
            break
    for experiment_id in experiment_id_list:
        nni_config = Config(experiment_dict[experiment_id]['fileName'])
        platform = nni_config.get_config('experimentConfig').get('trainingServicePlatform')
        experiment_id = nni_config.get_config('experimentId')
        if platform == 'remote':
            machine_list = nni_config.get_config('experimentConfig').get('machineList')
            remote_clean(machine_list, experiment_id)
        elif platform == 'pai':
            host = nni_config.get_config('experimentConfig').get('paiConfig').get('host')
            user_name = nni_config.get_config('experimentConfig').get('paiConfig').get('userName')
            output_dir = nni_config.get_config('experimentConfig').get('trial').get('outputDir')
            hdfs_clean(host, user_name, output_dir, experiment_id)
        elif platform != 'local':
            #TODO: support all platforms
            print_warning('platform {0} clean up not supported yet.'.format(platform))
            exit(0)
        #clean local data
        home = str(Path.home())
        local_dir = nni_config.get_config('experimentConfig').get('logDir')
        if not local_dir:
            local_dir = os.path.join(home, 'nni', 'experiments', experiment_id)
        local_clean(local_dir)
        experiment_config = Experiments()
        print_normal('removing metadata of experiment {0}'.format(experiment_id))
        experiment_config.remove_experiment(experiment_id)
        print_normal('Done.')

def get_platform_dir(config_content):
    '''get the dir list to be deleted'''
    platform = config_content.get('trainingServicePlatform')
    dir_list = []
    if platform == 'remote':
        machine_list = config_content.get('machineList')
        for machine in machine_list:
            host = machine.get('ip')
            port = machine.get('port')
            dir_list.append(host + ':' + str(port) + '/tmp/nni')
    elif platform == 'pai':
        host = config_content.get('paiConfig').get('host')
        user_name = config_content.get('paiConfig').get('userName')
        output_dir = config_content.get('trial').get('outputDir')
        dir_list.append('server: {0}, path: {1}/nni'.format(host, user_name))
        if output_dir:
            dir_list.append(output_dir)
    return dir_list

def platform_clean(args):
    '''clean up the experiment data'''
    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        print_error('Please set correct config path.')
        exit(1)
    config_content = get_yml_content(config_path)
    platform = config_content.get('trainingServicePlatform')
    if platform == 'local':
        print_normal('it doesn’t need to clean local platform.')
        exit(0)
    if platform not in ['remote', 'pai']:
        print_normal('platform {0} not supported.'.format(platform))
        exit(0)
    update_experiment()
    dir_list = get_platform_dir(config_content)
    if not dir_list:
        print_normal('No folder of NNI caches is found.')
        exit(1)
    while True:
        print_normal('This command will remove below folders of NNI caches. If other users are using experiments' \
                     ' on below hosts, it will be broken.')
        for value in dir_list:
            print('       ' + value)
        inputs = input('INFO: do you want to continue?[y/N]:')
        if not inputs.lower() or inputs.lower() in ['n', 'no']:
            exit(0)
        elif inputs.lower() not in ['y', 'n', 'yes', 'no']:
            print_warning('please input Y or N.')
        else:
            break
    if platform == 'remote':
        machine_list = config_content.get('machineList')
        remote_clean(machine_list, None)
    elif platform == 'pai':
        host = config_content.get('paiConfig').get('host')
        user_name = config_content.get('paiConfig').get('userName')
        output_dir = config_content.get('trial').get('outputDir')
        hdfs_clean(host, user_name, output_dir, None)
    print_normal('Done.')

def experiment_list(args):
    '''get the information of all experiments'''
    update_experiment()
    experiment_config = Experiments()
    experiment_dict = experiment_config.get_all_experiments()
    if not experiment_dict:
        print_normal('Cannot find experiments.')
        exit(1)
    experiment_id_list = []
    if args.all:
        for key in experiment_dict.keys():
            experiment_id_list.append(key)
    else:
        for key in experiment_dict.keys():
            if experiment_dict[key]['status'] != 'STOPPED':
                experiment_id_list.append(key)
        if not experiment_id_list:
            print_warning('There is no experiment running...\nYou can use \'nnictl experiment list --all\' to list all experiments.')
    experiment_information = ""
    for key in experiment_id_list:
        experiment_information += EXPERIMENT_DETAIL_FORMAT % (key,
                                                              experiment_dict[key].get('experimentName', 'N/A'),
                                                              experiment_dict[key]['status'],
                                                              experiment_dict[key]['port'],
                                                              experiment_dict[key].get('platform'),
                                                              experiment_dict[key]['startTime'],
                                                              experiment_dict[key]['endTime'])
    print(EXPERIMENT_INFORMATION_FORMAT % experiment_information)

def get_time_interval(time1, time2):
    '''get the interval of two times'''
    try:
        #convert time to timestamp
        time1 = time.mktime(time.strptime(time1, '%Y/%m/%d %H:%M:%S'))
        time2 = time.mktime(time.strptime(time2, '%Y/%m/%d %H:%M:%S'))
        seconds = (datetime.fromtimestamp(time2) - datetime.fromtimestamp(time1)).seconds
        #convert seconds to day:hour:minute:second
        days = seconds / 86400
        seconds %= 86400
        hours = seconds / 3600
        seconds %= 3600
        minutes = seconds / 60
        seconds %= 60
        return '%dd %dh %dm %ds' % (days, hours, minutes, seconds)
    except:
        return 'N/A'

def show_experiment_info():
    '''show experiment information in monitor'''
    update_experiment()
    experiment_config = Experiments()
    experiment_dict = experiment_config.get_all_experiments()
    if not experiment_dict:
        print('There is no experiment running...')
        exit(1)
    experiment_id_list = []
    for key in experiment_dict.keys():
        if experiment_dict[key]['status'] != 'STOPPED':
            experiment_id_list.append(key)
    if not experiment_id_list:
        print_warning('There is no experiment running...')
        return
    for key in experiment_id_list:
        print(EXPERIMENT_MONITOR_INFO % (key, experiment_dict[key]['status'], experiment_dict[key]['port'], \
             experiment_dict[key].get('platform'), experiment_dict[key]['startTime'], \
             get_time_interval(experiment_dict[key]['startTime'], experiment_dict[key]['endTime'])))
        print(TRIAL_MONITOR_HEAD)
        running, response = check_rest_server_quick(experiment_dict[key]['port'])
        if running:
            response = rest_get(trial_jobs_url(experiment_dict[key]['port']), REST_TIME_OUT)
            if response and check_response(response):
                content = json.loads(response.text)
                for index, value in enumerate(content):
                    content[index] = convert_time_stamp_to_date(value)
                    print(TRIAL_MONITOR_CONTENT % (content[index].get('id'), content[index].get('startTime'), \
                          content[index].get('endTime'), content[index].get('status')))
        print(TRIAL_MONITOR_TAIL)

def set_monitor(auto_exit, time_interval, port=None, pid=None):
    '''set the experiment monitor engine'''
    while True:
        try:
            if sys.platform == 'win32':
                os.system('cls')
            else:
                os.system('clear')
            update_experiment()
            show_experiment_info()
            if auto_exit:
                status = get_experiment_status(port)
                if status in ['DONE', 'ERROR', 'STOPPED']:
                    print_normal('Experiment status is {0}.'.format(status))
                    print_normal('Stopping experiment...')
                    kill_command(pid)
                    print_normal('Stop experiment success.')
                    exit(0)
            time.sleep(time_interval)
        except KeyboardInterrupt:
            if auto_exit:
                print_normal('Stopping experiment...')
                kill_command(pid)
                print_normal('Stop experiment success.')
            else:
                print_normal('Exiting...')
            exit(0)
        except Exception as exception:
            print_error(exception)
            exit(1)

def monitor_experiment(args):
    '''monitor the experiment'''
    if args.time <= 0:
        print_error('please input a positive integer as time interval, the unit is second.')
        exit(1)
    set_monitor(False, args.time)

def export_trials_data(args):
    '''export experiment metadata to csv
    '''
    nni_config = Config(get_config_filename(args))
    rest_port = nni_config.get_config('restServerPort')
    rest_pid = nni_config.get_config('restServerPid')
    if not detect_process(rest_pid):
        print_error('Experiment is not running...')
        return
    running, response = check_rest_server_quick(rest_port)
    if running:
        response = rest_get(export_data_url(rest_port), 20)
        if response is not None and check_response(response):
            if args.type == 'json':
                with open(args.path, 'w') as file:
                    file.write(response.text)
            elif args.type == 'csv':
                content = json.loads(response.text)
                trial_records = []
                for record in content:
                    record_value = json.loads(record['value'])
                    if not isinstance(record_value, (float, int)):
                        formated_record = {**record['parameter'], **record_value, **{'id': record['id']}}
                    else:
                        formated_record = {**record['parameter'], **{'reward': record_value, 'id': record['id']}}
                    trial_records.append(formated_record)
                if not trial_records:
                    print_error('No trial results collected! Please check your trial log...')
                    exit(0)
                with open(args.path, 'w', newline='') as file:
                    writer = csv.DictWriter(file, set.union(*[set(r.keys()) for r in trial_records]))
                    writer.writeheader()
                    writer.writerows(trial_records)
            else:
                print_error('Unknown type: %s' % args.type)
                exit(1)
        else:
            print_error('Export failed...')
    else:
        print_error('Restful server is not Running')

def search_space_auto_gen(args):
    '''dry run trial code to generate search space file'''
    trial_dir = os.path.expanduser(args.trial_dir)
    file_path = os.path.expanduser(args.file)
    if not os.path.isabs(file_path):
        file_path = os.path.join(os.getcwd(), file_path)
    assert os.path.exists(trial_dir)
    if os.path.exists(file_path):
        print_warning('%s already exists, will be overwritten.' % file_path)
    print_normal('Dry run to generate search space...')
    Popen(args.trial_command, cwd=trial_dir, env=dict(os.environ, NNI_GEN_SEARCH_SPACE=file_path), shell=True).wait()
    if not os.path.exists(file_path):
        print_warning('Expected search space file \'{}\' generated, but not found.'.format(file_path))
    else:
        print_normal('Generate search space done: \'{}\'.'.format(file_path))
