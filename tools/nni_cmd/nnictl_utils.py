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

import csv
import os
import psutil
import json
import datetime
import time
from subprocess import call, check_output
from nni_annotation import expand_annotations
from .rest_utils import rest_get, rest_delete, check_rest_server_quick, check_response
from .url_utils import trial_jobs_url, experiment_url, trial_job_id_url, export_data_url
from .config_utils import Config, Experiments
from .constants import NNICTL_HOME_DIR, EXPERIMENT_INFORMATION_FORMAT, EXPERIMENT_DETAIL_FORMAT, \
     EXPERIMENT_MONITOR_INFO, TRIAL_MONITOR_HEAD, TRIAL_MONITOR_CONTENT, TRIAL_MONITOR_TAIL, REST_TIME_OUT
from .common_utils import print_normal, print_error, print_warning, detect_process
from .command_utils import check_output_command, kill_command

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

def check_experiment_id(args):
    '''check if the id is valid
    '''
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
                experiment_information += (EXPERIMENT_DETAIL_FORMAT % (key, experiment_dict[key]['status'], \
                experiment_dict[key]['port'], experiment_dict[key].get('platform'), experiment_dict[key]['startTime'], experiment_dict[key]['endTime']))
            print(EXPERIMENT_INFORMATION_FORMAT % experiment_information)
            exit(1)
        elif not running_experiment_list:
            print_error('There is no experiment running!')
            return None
        else:
            return running_experiment_list[0]
    if experiment_dict.get(args.id):
        return args.id
    else:
        print_error('Id not correct!')
        return None

def parse_ids(args):
    '''Parse the arguments for nnictl stop
    1.If there is an id specified, return the corresponding id
    2.If there is no id specified, and there is an experiment running, return the id, or return Error
    3.If the id matches an experiment, nnictl will return the id.
    4.If the id ends with *, nnictl will match all ids matchs the regular
    5.If the id does not exist but match the prefix of an experiment id, nnictl will return the matched id
    6.If the id does not exist but match multiple prefix of the experiment ids, nnictl will give id information
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
    if not args.id:
        if len(running_experiment_list) > 1:
            print_error('There are multiple experiments, please set the experiment id...')
            experiment_information = ""
            for key in running_experiment_list:
                experiment_information += (EXPERIMENT_DETAIL_FORMAT % (key, experiment_dict[key]['status'], \
                experiment_dict[key]['port'], experiment_dict[key].get('platform'), experiment_dict[key]['startTime'], experiment_dict[key]['endTime']))
            print(EXPERIMENT_INFORMATION_FORMAT % experiment_information)
            exit(1)
        else:
            result_list = running_experiment_list
    elif args.id == 'all':
        result_list = running_experiment_list
    elif args.id.endswith('*'):
        for id in running_experiment_list:
            if id.startswith(args.id[:-1]):
                result_list.append(id)
    elif args.id in running_experiment_list:
        result_list.append(args.id)
    else:
        for id in running_experiment_list:
            if id.startswith(args.id):
                result_list.append(id)
        if len(result_list) > 1:
            print_error(args.id + ' is ambiguous, please choose ' + ' '.join(result_list) )
            return None
    if not result_list and args.id:
        print_error('There are no experiments matched, please set correct experiment id...')
    elif not result_list:
        print_error('There is no experiment running...')
    return result_list

def get_config_filename(args):
    '''get the file name of config file'''
    experiment_id = check_experiment_id(args)
    if experiment_id is None:
        print_error('Please set the experiment id!')
        exit(1)
    experiment_config = Experiments()
    experiment_dict = experiment_config.get_all_experiments()
    return experiment_dict[experiment_id]['fileName']

def get_experiment_port(args):
    '''get the port of experiment'''
    experiment_id = check_experiment_id(args)
    if experiment_id is None:
        print_error('Please set the experiment id!')
        exit(1)
    experiment_config = Experiments()
    experiment_dict = experiment_config.get_all_experiments()
    return experiment_dict[experiment_id]['port']

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
    nni_config = Config(get_config_filename(args))
    rest_port = nni_config.get_config('restServerPort')
    running, _ = check_rest_server_quick(rest_port)
    if not running:
        print_normal('Restful server is running...')
    else:
        print_normal('Restful server is not running...')

def stop_experiment(args):
    '''Stop the experiment which is running'''
    experiment_id_list = parse_ids(args)
    if experiment_id_list:
        experiment_config = Experiments()
        experiment_dict = experiment_config.get_all_experiments()
        for experiment_id in experiment_id_list:
            print_normal('Stoping experiment %s' % experiment_id)
            nni_config = Config(experiment_dict[experiment_id]['fileName'])
            rest_port = nni_config.get_config('restServerPort')
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
            print_normal('Stop experiment success!')
            experiment_config.update_experiment(experiment_id, 'status', 'STOPPED')
            time_now = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
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
                trial_id_path_dict[trial['id']] = trial['logPath']
    else:
        print_error('Restful server is not running...')
        exit(1)
    if args.id:
        if args.trial_id:
            if trial_id_path_dict.get(args.trial_id):
                print_normal('id:' + args.trial_id + ' path:' + trial_id_path_dict[args.trial_id])
            else:
                print_error('trial id is not valid!')
                exit(1)
        else:
            print_error('please specific the trial id!')
            exit(1)
    else:
        for key in trial_id_path_dict:
            print('id:' + key + ' path:' + trial_id_path_dict[key])

def get_config(args):
    '''get config info'''
    nni_config = Config(get_config_filename(args))
    print(nni_config.get_all_config())

def webui_url(args):
    '''show the url of web ui'''
    nni_config = Config(get_config_filename(args))
    print_normal('{0} {1}'.format('Web UI url:', ' '.join(nni_config.get_config('webuiUrl'))))

def experiment_list(args):
    '''get the information of all experiments'''
    experiment_config = Experiments()
    experiment_dict = experiment_config.get_all_experiments()
    if not experiment_dict:
        print('There is no experiment running...')
        exit(1)
    update_experiment()
    experiment_id_list = []
    if args.all and args.all == 'all':
        for key in experiment_dict.keys():
            experiment_id_list.append(key)
    else:
        for key in experiment_dict.keys():
            if experiment_dict[key]['status'] != 'STOPPED':
                experiment_id_list.append(key)
        if not experiment_id_list:
            print_warning('There is no experiment running...\nYou can use \'nnictl experiment list all\' to list all stopped experiments!')
    experiment_information = ""
    for key in experiment_id_list:
        
        experiment_information += (EXPERIMENT_DETAIL_FORMAT % (key, experiment_dict[key]['status'], experiment_dict[key]['port'],\
        experiment_dict[key].get('platform'), experiment_dict[key]['startTime'], experiment_dict[key]['endTime']))
    print(EXPERIMENT_INFORMATION_FORMAT % experiment_information)

def get_time_interval(time1, time2):
    '''get the interval of two times'''
    try:
        #convert time to timestamp
        time1 = time.mktime(time.strptime(time1, '%Y/%m/%d %H:%M:%S'))
        time2 = time.mktime(time.strptime(time2, '%Y/%m/%d %H:%M:%S'))
        seconds = (datetime.datetime.fromtimestamp(time2) - datetime.datetime.fromtimestamp(time1)).seconds
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
    experiment_config = Experiments()
    experiment_dict = experiment_config.get_all_experiments()
    if not experiment_dict:
        print('There is no experiment running...')
        exit(1)
    update_experiment()
    experiment_id_list = []
    for key in experiment_dict.keys():
        if experiment_dict[key]['status'] != 'STOPPED':
            experiment_id_list.append(key)
    if not experiment_id_list:
        print_warning('There is no experiment running...')
        return
    for key in experiment_id_list:
        print(EXPERIMENT_MONITOR_INFO % (key, experiment_dict[key]['status'], experiment_dict[key]['port'], \
             experiment_dict[key].get('platform'), experiment_dict[key]['startTime'], get_time_interval(experiment_dict[key]['startTime'], experiment_dict[key]['endTime'])))
        print(TRIAL_MONITOR_HEAD)
        running, response = check_rest_server_quick(experiment_dict[key]['port'])
        if running:
            response = rest_get(trial_jobs_url(experiment_dict[key]['port']), REST_TIME_OUT)
            if response and check_response(response):
                content = json.loads(response.text)
                for index, value in enumerate(content):
                    content[index] = convert_time_stamp_to_date(value)
                    print(TRIAL_MONITOR_CONTENT % (content[index].get('id'), content[index].get('startTime'), content[index].get('endTime'), content[index].get('status')))
        print(TRIAL_MONITOR_TAIL)

def monitor_experiment(args):
    '''monitor the experiment'''
    if args.time <= 0:
        print_error('please input a positive integer as time interval, the unit is second.')
        exit(1)
    while True:
        try:
            os.system('clear')
            update_experiment()
            show_experiment_info()
            time.sleep(args.time)
        except KeyboardInterrupt:
            exit(0)
        except Exception as exception:
            print_error(exception)
            exit(1)

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
                    if not isinstance(record['value'], (float, int)):
                        formated_record = {**record['parameter'], **record['value'], **{'id': record['id']}}
                    else:
                        formated_record = {**record['parameter'], **{'reward': record['value'], 'id': record['id']}}
                    trial_records.append(formated_record)
                with open(args.path, 'w') as file:
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