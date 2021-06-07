# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import csv
import os
import sys
import json
import time
import shutil
import subprocess
from functools import cmp_to_key
import traceback
from datetime import datetime, timezone
from subprocess import Popen
from nni.tools.annotation import expand_annotations
import nni_node  # pylint: disable=import-error
from .rest_utils import rest_get, rest_delete, check_rest_server_quick, check_response
from .url_utils import trial_jobs_url, experiment_url, trial_job_id_url, export_data_url, metric_data_url
from .config_utils import Config, Experiments
from .constants import NNI_HOME_DIR, EXPERIMENT_INFORMATION_FORMAT, EXPERIMENT_DETAIL_FORMAT, EXPERIMENT_MONITOR_INFO, \
     TRIAL_MONITOR_HEAD, TRIAL_MONITOR_CONTENT, TRIAL_MONITOR_TAIL, REST_TIME_OUT
from .common_utils import print_normal, print_error, print_warning, detect_process, get_yml_content, generate_temp_dir
from .common_utils import print_green
from .command_utils import check_output_command, kill_command
from .ssh_utils import create_ssh_sftp_client, remove_remote_directory

def get_experiment_time(port):
    '''get the startTime and endTime of an experiment'''
    response = rest_get(experiment_url(port), REST_TIME_OUT)
    if response and check_response(response):
        content = json.loads(response.text)
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
    experiments_config = Experiments()
    experiments_dict = experiments_config.get_all_experiments()
    if not experiments_dict:
        return None
    for key in experiments_dict.keys():
        if isinstance(experiments_dict[key], dict):
            if experiments_dict[key].get('status') != 'STOPPED':
                rest_pid = experiments_dict[key].get('pid')
                if not detect_process(rest_pid):
                    experiments_config.update_experiment(key, 'status', 'STOPPED')
                    continue

def check_experiment_id(args, update=True):
    '''check if the id is valid
    '''
    if update:
        update_experiment()
    experiments_config = Experiments()
    experiments_dict = experiments_config.get_all_experiments()
    if not experiments_dict:
        print_normal('There is no experiment running...')
        return None
    if not args.id:
        running_experiment_list = []
        for key in experiments_dict.keys():
            if isinstance(experiments_dict[key], dict):
                if experiments_dict[key].get('status') != 'STOPPED':
                    running_experiment_list.append(key)
            elif isinstance(experiments_dict[key], list):
                # if the config file is old version, remove the configuration from file
                experiments_config.remove_experiment(key)
        if len(running_experiment_list) > 1:
            print_error('There are multiple experiments, please set the experiment id...')
            experiment_information = ""
            for key in running_experiment_list:
                experiment_information += EXPERIMENT_DETAIL_FORMAT % (key,
                                                                      experiments_dict[key].get('experimentName', 'N/A'),
                                                                      experiments_dict[key]['status'],
                                                                      experiments_dict[key].get('port', 'N/A'),
                                                                      experiments_dict[key].get('platform'),
                                                                      time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(experiments_dict[key]['startTime'] / 1000)) if isinstance(experiments_dict[key]['startTime'], int) else experiments_dict[key]['startTime'],
                                                                      time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(experiments_dict[key]['endTime'] / 1000)) if isinstance(experiments_dict[key]['endTime'], int) else experiments_dict[key]['endTime'])
            print(EXPERIMENT_INFORMATION_FORMAT % experiment_information)
            exit(1)
        elif not running_experiment_list:
            print_error('There is no experiment running.')
            return None
        else:
            return running_experiment_list[0]
    if experiments_dict.get(args.id):
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
    experiments_config = Experiments()
    experiments_dict = experiments_config.get_all_experiments()
    if not experiments_dict:
        print_normal('Experiment is not running...')
        return None
    result_list = []
    running_experiment_list = []
    for key in experiments_dict.keys():
        if isinstance(experiments_dict[key], dict):
            if experiments_dict[key].get('status') != 'STOPPED':
                running_experiment_list.append(key)
        elif isinstance(experiments_dict[key], list):
            # if the config file is old version, remove the configuration from file
            experiments_config.remove_experiment(key)
    if args.all:
        return running_experiment_list
    if args.port is not None:
        for key in running_experiment_list:
            if experiments_dict[key].get('port') == args.port:
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
                                                                      experiments_dict[key].get('experimentName', 'N/A'),
                                                                      experiments_dict[key]['status'],
                                                                      experiments_dict[key].get('port', 'N/A'),
                                                                      experiments_dict[key].get('platform'),
                                                                      time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(experiments_dict[key]['startTime'] / 1000)) if isinstance(experiments_dict[key]['startTime'], int) else experiments_dict[key]['startTime'],
                                                                      time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(experiments_dict[key]['endTime'] / 1000)) if isinstance(experiments_dict[key]['endTime'], int) else experiments_dict[key]['endTime'])
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
    return experiment_id

def get_experiment_port(args):
    '''get the port of experiment'''
    experiment_id = check_experiment_id(args)
    if experiment_id is None:
        print_error('Please set correct experiment id.')
        exit(1)
    experiments_config = Experiments()
    experiments_dict = experiments_config.get_all_experiments()
    return experiments_dict[experiment_id].get('port')

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
    experiments_config = Experiments()
    experiments_dict = experiments_config.get_all_experiments()
    rest_port = experiments_dict.get(get_config_filename(args)).get('port')
    running, _ = check_rest_server_quick(rest_port)
    if running:
        print_normal('Restful server is running...')
    else:
        print_normal('Restful server is not running...')
    return running

def stop_experiment(args):
    '''Stop the experiment which is running'''
    if args.id and args.id == 'all':
        print_warning('\'nnictl stop all\' is abolished, please use \'nnictl stop --all\' to stop all of experiments!')
        exit(1)
    experiment_id_list = parse_ids(args)
    if experiment_id_list:
        for experiment_id in experiment_id_list:
            print_normal('Stopping experiment %s' % experiment_id)
            experiments_config = Experiments()
            experiments_dict = experiments_config.get_all_experiments()
            rest_pid = experiments_dict.get(experiment_id).get('pid')
            if rest_pid:
                kill_command(rest_pid)
            print_normal('Stop experiment success.')

def trial_ls(args):
    '''List trial'''
    def final_metric_data_cmp(lhs, rhs):
        metric_l = json.loads(json.loads(lhs['finalMetricData'][0]['data']))
        metric_r = json.loads(json.loads(rhs['finalMetricData'][0]['data']))
        if isinstance(metric_l, float):
            return metric_l - metric_r
        elif isinstance(metric_l, dict):
            return metric_l['default'] - metric_r['default']
        else:
            print_error('Unexpected data format. Please check your data.')
            raise ValueError

    if args.head and args.tail:
        print_error('Head and tail cannot be set at the same time.')
        return
    experiments_config = Experiments()
    experiments_dict = experiments_config.get_all_experiments()
    experiment_id = get_config_filename(args)
    rest_port = experiments_dict.get(experiment_id).get('port')
    rest_pid = experiments_dict.get(experiment_id).get('pid')
    if not detect_process(rest_pid):
        print_error('Experiment is not running...')
        return
    running, response = check_rest_server_quick(rest_port)
    if running:
        response = rest_get(trial_jobs_url(rest_port), REST_TIME_OUT)
        if response and check_response(response):
            content = json.loads(response.text)
            if args.head:
                assert args.head > 0, 'The number of requested data must be greater than 0.'
                content = sorted(filter(lambda x: 'finalMetricData' in x, content),
                                 key=cmp_to_key(final_metric_data_cmp), reverse=True)[:args.head]
            elif args.tail:
                assert args.tail > 0, 'The number of requested data must be greater than 0.'
                content = sorted(filter(lambda x: 'finalMetricData' in x, content),
                                 key=cmp_to_key(final_metric_data_cmp))[:args.tail]
            for index, value in enumerate(content):
                content[index] = convert_time_stamp_to_date(value)
            print(json.dumps(content, indent=4, sort_keys=True, separators=(',', ':')))
            return content
        else:
            print_error('List trial failed...')
    else:
        print_error('Restful server is not running...')
    return None

def trial_kill(args):
    '''List trial'''
    experiments_config = Experiments()
    experiments_dict = experiments_config.get_all_experiments()
    experiment_id = get_config_filename(args)
    rest_port = experiments_dict.get(experiment_id).get('port')
    rest_pid = experiments_dict.get(experiment_id).get('pid')
    if not detect_process(rest_pid):
        print_error('Experiment is not running...')
        return
    running, _ = check_rest_server_quick(rest_port)
    if running:
        response = rest_delete(trial_job_id_url(rest_port, args.trial_id), REST_TIME_OUT)
        if response and check_response(response):
            print(response.text)
            return True
        else:
            print_error('Kill trial job failed...')
    else:
        print_error('Restful server is not running...')
    return False

def trial_codegen(args):
    '''Generate code for a specific trial'''
    print_warning('Currently, this command is only for nni nas programming interface.')
    exp_id = get_config_filename(args)
    experiment_config = Config(exp_id, Experiments().get_all_experiments()[exp_id]['logDir']).get_config()
    if not experiment_config.get('useAnnotation'):
        print_error('The experiment is not using annotation')
        exit(1)
    code_dir = experiment_config['trial']['codeDir']
    expand_annotations(code_dir, './exp_%s_trial_%s_code'%(exp_id, args.trial_id), exp_id, args.trial_id)

def list_experiment(args):
    '''Get experiment information'''
    experiments_config = Experiments()
    experiments_dict = experiments_config.get_all_experiments()
    experiment_id = get_config_filename(args)
    rest_port = experiments_dict.get(experiment_id).get('port')
    rest_pid = experiments_dict.get(experiment_id).get('pid')
    if not detect_process(rest_pid):
        print_error('Experiment is not running...')
        return
    running, _ = check_rest_server_quick(rest_port)
    if running:
        response = rest_get(experiment_url(rest_port), REST_TIME_OUT)
        if response and check_response(response):
            content = convert_time_stamp_to_date(json.loads(response.text))
            print(json.dumps(content, indent=4, sort_keys=True, separators=(',', ':')))
            return content
        else:
            print_error('List experiment failed...')
    else:
        print_error('Restful server is not running...')
    return None

def experiment_status(args):
    '''Show the status of experiment'''
    experiments_config = Experiments()
    experiments_dict = experiments_config.get_all_experiments()
    rest_port = experiments_dict.get(get_config_filename(args)).get('port')
    result, response = check_rest_server_quick(rest_port)
    if not result:
        print_normal('Restful server is not running...')
    else:
        print(json.dumps(json.loads(response.text), indent=4, sort_keys=True, separators=(',', ':')))
    return result

def log_internal(args, filetype):
    '''internal function to call get_log_content'''
    file_name = get_config_filename(args)
    if filetype == 'stdout':
        file_full_path = os.path.join(NNI_HOME_DIR, file_name, 'log', 'nnictl_stdout.log')
    else:
        file_full_path = os.path.join(NNI_HOME_DIR, file_name, 'log', 'nnictl_stderr.log')
    print(check_output_command(file_full_path, head=args.head, tail=args.tail))

def log_stdout(args):
    '''get stdout log'''
    log_internal(args, 'stdout')

def log_stderr(args):
    '''get stderr log'''
    log_internal(args, 'stderr')

def log_trial_adl_helper(args, experiment_id):
    # adljob_id format should be consistent to the one in "adlTrainingService.ts":
    #   const adlJobName: string = `nni-exp-${this.experimentId}-trial-${trialJobId}`.toLowerCase();
    adlJobName = "nni-exp-{}-trial-{}".format(experiment_id, args.trial_id).lower()
    print_warning('Note that no log will show when trial is pending or done (succeeded or failed). '
                  'You can retry the command.')
    print_green('>>> Trial log streaming:')
    try:
        subprocess.run(
            [
                "kubectl", "logs",
                "-l", "adaptdl/job=%s" % adlJobName,
                "-f"  # Follow the stream
            ],  # TODO: support remaining argument, uncomment the lines in nnictl.py
        )  # TODO: emulate tee behaviors, not necessary tho.
    except KeyboardInterrupt:
        pass
    except Exception:
        print_error('Error! Please check kubectl:')
        traceback.print_exc()
        exit(1)
    finally:
        print_green('<<< [adlJobName:%s]' % adlJobName)
        nni_manager_collection_path = os.path.expanduser('~/nni-experiments/%s/trials/%s/stdout_log_collection.log' %
                                                         (experiment_id, args.trial_id))
        print_green('>>> (Optional) How to persist the complete trial log locally:')
        print(
            'Please ensure `logCollection: http` '
            'exists in the experiment configuration yaml. '
            'After trial done, you can check it from the file below: \n  %s'
            % nni_manager_collection_path
        )


def log_trial(args):
    ''''get trial log path'''
    trial_id_path_dict = {}
    trial_id_list = []
    experiments_config = Experiments()
    experiments_dict = experiments_config.get_all_experiments()
    experiment_id = get_config_filename(args)
    rest_port = experiments_dict.get(experiment_id).get('port')
    rest_pid = experiments_dict.get(experiment_id).get('pid')
    experiment_config = Config(experiment_id, experiments_dict.get(experiment_id).get('logDir')).get_config()
    if not detect_process(rest_pid):
        print_error('Experiment is not running...')
        return
    running, response = check_rest_server_quick(rest_port)
    if running:
        response = rest_get(trial_jobs_url(rest_port), REST_TIME_OUT)
        if response and check_response(response):
            content = json.loads(response.text)
            for trial in content:
                trial_id_list.append(trial.get('trialJobId'))
                if trial.get('logPath'):
                    trial_id_path_dict[trial.get('trialJobId')] = trial['logPath']
    else:
        print_error('Restful server is not running...')
        exit(1)
    is_adl = experiment_config.get('trainingServicePlatform') == 'adl'
    if is_adl and not args.trial_id:
        print_error('Trial ID is required to retrieve the log for adl. Please specify it with "--trial_id".')
        exit(1)
    if args.trial_id:
        if args.trial_id not in trial_id_list:
            print_error('Trial id {0} not correct, please check your command!'.format(args.trial_id))
            exit(1)
        if is_adl:
            log_trial_adl_helper(args, experiment_id)
            # adl has its own way to log trial, and it thus returns right after the helper returns
            return
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
    experiment_id = get_config_filename(args)
    experiment_config = Config(experiment_id, Experiments().get_all_experiments()[experiment_id]['logDir']).get_config()
    print(json.dumps(experiment_config, indent=4))

def webui_url(args):
    '''show the url of web ui'''
    experiment_id = get_config_filename(args)
    experiments_dict = Experiments().get_all_experiments()
    print_normal('{0} {1}'.format('Web UI url:', ' '.join(experiments_dict[experiment_id].get('webuiUrl'))))

def webui_nas(args):
    '''launch nas ui'''
    print_normal('Starting NAS UI...')
    try:
        entry_dir = nni_node.__path__[0]
        entry_file = os.path.join(entry_dir, 'nasui', 'server.js')
        if sys.platform == 'win32':
            node_command = os.path.join(entry_dir, 'node.exe')
        else:
            node_command = os.path.join(entry_dir, 'node')
        cmds = [node_command, '--max-old-space-size=4096', entry_file, '--port', str(args.port), '--logdir', args.logdir]
        subprocess.run(cmds, cwd=entry_dir)
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
            remote_dir = '/' + '/'.join(['tmp', 'nni-experiments', experiment_id])
        else:
            remote_dir = '/' + '/'.join(['tmp', 'nni-experiments'])
        sftp = create_ssh_sftp_client(host, port, userName, passwd, sshKeyPath, passphrase)
        print_normal('removing folder {0}'.format(host + ':' + str(port) + remote_dir))
        remove_remote_directory(sftp, remote_dir)

def experiment_clean(args):
    '''clean up the experiment data'''
    experiment_id_list = []
    experiments_config = Experiments()
    experiments_dict = experiments_config.get_all_experiments()
    if args.all:
        experiment_id_list = list(experiments_dict.keys())
    else:
        if args.id is None:
            print_error('please set experiment id.')
            exit(1)
        if args.id not in experiments_dict:
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
        experiment_id = get_config_filename(args)
        experiment_config = Config(experiment_id, Experiments().get_all_experiments()[experiment_id]['logDir']).get_config()
        platform = experiment_config.get('trainingServicePlatform')
        if platform == 'remote':
            machine_list = experiment_config.get('machineList')
            remote_clean(machine_list, experiment_id)
        elif platform != 'local':
            # TODO: support all platforms
            print_warning('platform {0} clean up not supported yet.'.format(platform))
            exit(0)
        # clean local data
        local_base_dir = experiments_config.experiments[experiment_id]['logDir']
        if not local_base_dir:
            local_base_dir = NNI_HOME_DIR
        local_experiment_dir = os.path.join(local_base_dir, experiment_id)
        experiment_folder_name_list = ['checkpoint', 'db', 'log', 'trials']
        for folder_name in experiment_folder_name_list:
            local_clean(os.path.join(local_experiment_dir, folder_name))
        if not os.listdir(local_experiment_dir):
            local_clean(local_experiment_dir)
        print_normal('removing metadata of experiment {0}'.format(experiment_id))
        experiments_config.remove_experiment(experiment_id)
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
        remote_clean(machine_list)
    print_normal('Done.')

def experiment_list(args):
    '''get the information of all experiments'''
    update_experiment()
    experiments_config = Experiments()
    experiments_dict = experiments_config.get_all_experiments()
    if not experiments_dict:
        print_normal('Cannot find experiments.')
        exit(1)
    experiment_id_list = []
    if args.all:
        for key in experiments_dict.keys():
            experiment_id_list.append(key)
    else:
        for key in experiments_dict.keys():
            if experiments_dict[key]['status'] != 'STOPPED':
                experiment_id_list.append(key)
        if not experiment_id_list:
            print_warning('There is no experiment running...\nYou can use \'nnictl experiment list --all\' to list all experiments.')
    experiment_information = ""
    for key in experiment_id_list:
        experiment_information += EXPERIMENT_DETAIL_FORMAT % (key,
                                                              experiments_dict[key].get('experimentName', 'N/A'),
                                                              experiments_dict[key]['status'],
                                                              experiments_dict[key].get('port', 'N/A'),
                                                              experiments_dict[key].get('platform'),
                                                              time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(experiments_dict[key]['startTime'] / 1000)) if isinstance(experiments_dict[key]['startTime'], int) else experiments_dict[key]['startTime'],
                                                              time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(experiments_dict[key]['endTime'] / 1000)) if isinstance(experiments_dict[key]['endTime'], int) else experiments_dict[key]['endTime'])
    print(EXPERIMENT_INFORMATION_FORMAT % experiment_information)
    return experiment_id_list

def get_time_interval(time1, time2):
    '''get the interval of two times'''
    try:
        seconds = int((time2 - time1) / 1000)
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
    experiments_config = Experiments()
    experiments_dict = experiments_config.get_all_experiments()
    if not experiments_dict:
        print('There is no experiment running...')
        exit(1)
    experiment_id_list = []
    for key in experiments_dict.keys():
        if experiments_dict[key]['status'] != 'STOPPED':
            experiment_id_list.append(key)
    if not experiment_id_list:
        print_warning('There is no experiment running...')
        return
    for key in experiment_id_list:
        print(EXPERIMENT_MONITOR_INFO % (key, experiments_dict[key]['status'], experiments_dict[key]['port'], \
              experiments_dict[key].get('platform'), time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(experiments_dict[key]['startTime'] / 1000)) if isinstance(experiments_dict[key]['startTime'], int) else experiments_dict[key]['startTime'], \
              get_time_interval(experiments_dict[key]['startTime'], experiments_dict[key]['endTime'])))
        print(TRIAL_MONITOR_HEAD)
        running, response = check_rest_server_quick(experiments_dict[key]['port'])
        if running:
            response = rest_get(trial_jobs_url(experiments_dict[key]['port']), REST_TIME_OUT)
            if response and check_response(response):
                content = json.loads(response.text)
                for index, value in enumerate(content):
                    content[index] = convert_time_stamp_to_date(value)
                    print(TRIAL_MONITOR_CONTENT % (content[index].get('trialJobId'), content[index].get('startTime'), \
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
    '''export experiment metadata and intermediate results to json or csv
    '''
    def groupby_trial_id(intermediate_results):
        sorted(intermediate_results, key=lambda x: x['timestamp'])
        groupby = dict()
        for content in intermediate_results:
            groupby.setdefault(content['trialJobId'], []).append(json.loads(content['data']))
        return groupby

    experiments_config = Experiments()
    experiments_dict = experiments_config.get_all_experiments()
    experiment_id = get_config_filename(args)
    rest_port = experiments_dict.get(experiment_id).get('port')
    rest_pid = experiments_dict.get(experiment_id).get('pid')

    if not detect_process(rest_pid):
        print_error('Experiment is not running...')
        return
    running, response = check_rest_server_quick(rest_port)
    if not running:
        print_error('Restful server is not running')
        return
    response = rest_get(export_data_url(rest_port), 20)
    if response is not None and check_response(response):
        content = json.loads(response.text)
        if args.intermediate:
            intermediate_results_response = rest_get(metric_data_url(rest_port), REST_TIME_OUT)
            if not intermediate_results_response or not check_response(intermediate_results_response):
                print_error('Error getting intermediate results.')
                return
            intermediate_results = groupby_trial_id(json.loads(intermediate_results_response.text))
            for record in content:
                record['intermediate'] = intermediate_results[record['trialJobId']]
        if args.type == 'json':
            with open(args.path, 'w') as file:
                file.write(json.dumps(content))
        elif args.type == 'csv':
            trial_records = []
            for record in content:
                formated_record = dict()
                if args.intermediate:
                    formated_record['intermediate'] = '[' + ','.join(record['intermediate']) + ']'
                record_value = json.loads(record['value'])
                if not isinstance(record_value, (float, int)):
                    formated_record.update({**record['parameter'], **record_value, **{'trialJobId': record['trialJobId']}})
                else:
                    formated_record.update({**record['parameter'], **{'reward': record_value, 'trialJobId': record['trialJobId']}})
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
            return
    else:
        print_error('Export failed...')

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

def save_experiment(args):
    '''save experiment data to a zip file'''
    experiments_config = Experiments()
    experiments_dict = experiments_config.get_all_experiments()
    if args.id is None:
        print_error('Please set experiment id.')
        exit(1)
    if args.id not in experiments_dict:
        print_error('Cannot find experiment {0}.'.format(args.id))
        exit(1)
    if experiments_dict[args.id].get('status') != 'STOPPED':
        print_error('Can only save stopped experiment!')
        exit(1)
    print_normal('Saving...')
    experiment_config = Config(args.id, experiments_dict[args.id]['logDir']).get_config()
    logDir = os.path.join(experiments_dict[args.id]['logDir'], args.id)
    temp_root_dir = generate_temp_dir()

    # Step1. Copy logDir to temp folder
    if not os.path.exists(logDir):
        print_error('logDir: %s does not exist!' % logDir)
        exit(1)
    temp_experiment_dir = os.path.join(temp_root_dir, 'experiment')
    shutil.copytree(logDir, temp_experiment_dir)

    # Step2. Copy nnictl metadata to temp folder
    temp_nnictl_dir = os.path.join(temp_root_dir, 'nnictl')
    os.makedirs(temp_nnictl_dir, exist_ok=True)
    try:
        with open(os.path.join(temp_nnictl_dir, '.experiment'), 'w') as file:
            experiments_dict[args.id]['id'] = args.id
            json.dump(experiments_dict[args.id], file)
    except IOError:
        print_error('Write file to %s failed!' % os.path.join(temp_nnictl_dir, '.experiment'))
        exit(1)
    nnictl_log_dir = os.path.join(NNI_HOME_DIR, args.id, 'log')
    shutil.copytree(nnictl_log_dir, os.path.join(temp_nnictl_dir, args.id, 'log'))

    # Step3. Copy code dir
    if args.saveCodeDir:
        temp_code_dir = os.path.join(temp_root_dir, 'code')
        shutil.copytree(experiment_config['trial']['codeDir'], temp_code_dir)

    # Step4. Copy searchSpace file
    search_space_path = experiment_config.get('searchSpacePath')
    if search_space_path:
        if not os.path.exists(search_space_path):
            print_warning('search space %s does not exist!' % search_space_path)
        else:
            temp_search_space_dir = os.path.join(temp_root_dir, 'searchSpace')
            os.makedirs(temp_search_space_dir, exist_ok=True)
            search_space_name = os.path.basename(search_space_path)
            shutil.copyfile(search_space_path, os.path.join(temp_search_space_dir, search_space_name))

    # Step5. Archive folder
    zip_package_name = 'nni_experiment_%s' % args.id
    if args.path:
        os.makedirs(args.path, exist_ok=True)
        zip_package_name = os.path.join(args.path, zip_package_name)
    shutil.make_archive(zip_package_name, 'zip', temp_root_dir)
    print_normal('Save to %s.zip success!' % zip_package_name)

    # Step5. Cleanup temp data
    shutil.rmtree(temp_root_dir)

def load_experiment(args):
    '''load experiment data'''
    package_path = os.path.expanduser(args.path)
    if not os.path.exists(args.path):
        print_error('file path %s does not exist!' % args.path)
        exit(1)
    if args.searchSpacePath and os.path.isdir(args.searchSpacePath):
        print_error('search space path should be a full path with filename, not a directory!')
        exit(1)
    temp_root_dir = generate_temp_dir()
    shutil.unpack_archive(package_path, temp_root_dir)
    print_normal('Loading...')
    # Step1. Validation
    if not os.path.exists(args.codeDir):
        print_error('Invalid: codeDir path does not exist!')
        exit(1)
    if args.logDir:
        if not os.path.exists(args.logDir):
            print_error('Invalid: logDir path does not exist!')
            exit(1)
    experiment_temp_dir = os.path.join(temp_root_dir, 'experiment')
    if not os.path.exists(os.path.join(experiment_temp_dir, 'db')):
        print_error('Invalid archive file: db file does not exist!')
        shutil.rmtree(temp_root_dir)
        exit(1)
    nnictl_temp_dir = os.path.join(temp_root_dir, 'nnictl')
    if not os.path.exists(os.path.join(nnictl_temp_dir, '.experiment')):
        print_error('Invalid archive file: nnictl metadata file does not exist!')
        shutil.rmtree(temp_root_dir)
        exit(1)
    try:
        with open(os.path.join(nnictl_temp_dir, '.experiment'), 'r') as file:
            experiment_metadata = json.load(file)
    except ValueError as err:
        print_error('Invalid nnictl metadata file: %s' % err)
        shutil.rmtree(temp_root_dir)
        exit(1)
    experiments_config = Experiments()
    experiments_dict = experiments_config.get_all_experiments()
    experiment_id = experiment_metadata.get('id')
    if experiment_id in experiments_dict:
        print_error('Invalid: experiment id already exist!')
        shutil.rmtree(temp_root_dir)
        exit(1)
    if not os.path.exists(os.path.join(nnictl_temp_dir, experiment_id)):
        print_error('Invalid: experiment metadata does not exist!')
        shutil.rmtree(temp_root_dir)
        exit(1)

    # Step2. Copy nnictl metadata
    src_path = os.path.join(nnictl_temp_dir, experiment_id)
    dest_path = os.path.join(NNI_HOME_DIR, experiment_id)
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)
    shutil.copytree(src_path, dest_path)

    # Step3. Copy experiment data
    os.rename(os.path.join(temp_root_dir, 'experiment'), os.path.join(temp_root_dir, experiment_id))
    src_path = os.path.join(os.path.join(temp_root_dir, experiment_id))
    experiment_config = Config(experiment_id, temp_root_dir).get_config()
    if args.logDir:
        logDir = args.logDir
        experiment_config['logDir'] = logDir
    else:
        if experiment_config.get('logDir'):
            logDir = experiment_config['logDir']
        else:
            logDir = NNI_HOME_DIR

    dest_path = os.path.join(logDir, experiment_id)
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)
    shutil.copytree(src_path, dest_path)

    # Step4. Copy code dir
    codeDir = os.path.expanduser(args.codeDir)
    if not os.path.isabs(codeDir):
        codeDir = os.path.join(os.getcwd(), codeDir)
        print_normal('Expand codeDir to %s' % codeDir)
    experiment_config['trial']['codeDir'] = codeDir
    archive_code_dir = os.path.join(temp_root_dir, 'code')
    if os.path.exists(archive_code_dir):
        file_list = os.listdir(archive_code_dir)
        for file_name in file_list:
            src_path = os.path.join(archive_code_dir, file_name)
            target_path = os.path.join(codeDir, file_name)
            if os.path.exists(target_path):
                print_error('Copy %s failed, %s exist!' % (file_name, target_path))
                continue
            if os.path.isdir(src_path):
                shutil.copytree(src_path, target_path)
            else:
                shutil.copy(src_path, target_path)

    # Step5. Create experiment metadata
    experiments_config.add_experiment(experiment_id,
                                      experiment_metadata.get('port'),
                                      experiment_metadata.get('startTime'),
                                      experiment_metadata.get('platform'),
                                      experiment_metadata.get('experimentName'),
                                      experiment_metadata.get('endTime'),
                                      experiment_metadata.get('status'),
                                      experiment_metadata.get('tag'),
                                      experiment_metadata.get('pid'),
                                      experiment_metadata.get('webUrl'),
                                      logDir)
    print_normal('Load experiment %s succsss!' % experiment_id)

    # Step6. Cleanup temp data
    shutil.rmtree(temp_root_dir)
