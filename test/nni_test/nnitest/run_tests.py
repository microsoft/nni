# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import argparse
import subprocess
import time
import traceback
import json
import torch
import ruamel.yaml as yaml

from utils import setup_experiment, get_experiment_status, get_yml_content, dump_yml_content, get_experiment_id, \
    parse_max_duration_time, get_succeeded_trial_num, deep_update, print_trial_job_log, get_failed_trial_jobs
from utils import GREEN, RED, CLEAR, STATUS_URL, TRIAL_JOBS_URL, EXPERIMENT_URL, REST_ENDPOINT
import validators

it_variables = {}

def update_training_service_config(config, training_service):
    it_ts_config = get_yml_content(os.path.join('config', 'training_service.yml'))

    # hack for kubeflow trial config
    if training_service == 'kubeflow':
        it_ts_config[training_service]['trial']['worker']['command'] = config['trial']['command']
        config['trial'].pop('command')
        if 'gpuNum' in config['trial']:
            config['trial'].pop('gpuNum')

    if training_service == 'frameworkcontroller':
        it_ts_config[training_service]['trial']['taskRoles'][0]['command'] = config['trial']['command']
        config['trial'].pop('command')
        if 'gpuNum' in config['trial']:
            config['trial'].pop('gpuNum')

    deep_update(config, it_ts_config['all'])
    deep_update(config, it_ts_config[training_service])

def run_test_case(test_case_config, it_config, args):
    # fill test case default config
    for k in it_config['defaultTestCaseConfig']:
        if k not in test_case_config:
            test_case_config[k] = it_config['defaultTestCaseConfig'][k]
    print(json.dumps(test_case_config, indent=4))

    config_path = os.path.join(args.nni_source_dir, test_case_config['configFile'])
    test_yml_config = get_yml_content(config_path)

    # apply training service config
    update_training_service_config(test_yml_config, args.ts)

    # apply test case specific config
    if test_case_config.get('config') is not None:
        deep_update(test_yml_config, test_case_config['config'])

    # check GPU
    if test_yml_config['trial']['gpuNum'] > 0 and torch.cuda.device_count() < 1:
        print('skipping {}, gpu is not available'.format(test_case_config['name']))
        return

    # generate temporary config yml file to launch experiment
    new_config_file = config_path + '.tmp'
    dump_yml_content(new_config_file, test_yml_config)
    print(yaml.dump(test_yml_config, default_flow_style=False))

    # set configFile variable
    it_variables['$configFile'] = new_config_file

    try:
        launch_test(new_config_file, args.ts, test_case_config)

        validator_name = test_case_config.get('validator')
        if validator_name is not None:
            validator = validators.__dict__[validator_name]()
            validator(REST_ENDPOINT, None, args.nni_source_dir)
    finally:
        print('Stop command:', test_case_config.get('stopCommand'))
        if test_case_config.get('stopCommand'):
            subprocess.run(test_case_config.get('stopCommand').split(' '))
        # remove tmp config file
        if os.path.exists(new_config_file):
            os.remove(new_config_file)

def get_max_values(config_file):
    '''Get maxExecDuration and maxTrialNum of experiment'''
    experiment_config = get_yml_content(config_file)
    return parse_max_duration_time(experiment_config['maxExecDuration']), experiment_config['maxTrialNum']

def get_launch_command(test_case_config):
    launch_command = test_case_config.get('launchCommand')
    assert launch_command is not None

    # replace variables
    for k in it_variables:
        launch_command = launch_command.replace(k, it_variables[k])
    print('launch command: ', launch_command)
    return launch_command

def launch_test(config_file, training_service, test_case_config):
    '''run test per configuration file'''

    proc = subprocess.run(get_launch_command(test_case_config).split(' '))
    assert proc.returncode == 0, '`nnictl create` failed with code %d' % proc.returncode

    # set experiment ID into variable
    exp_var_name = test_case_config.get('setExperimentIdtoVar')
    if exp_var_name is not None:
        assert exp_var_name.startswith('$')
        it_variables[exp_var_name] = get_experiment_id(EXPERIMENT_URL)
    print('variables:', it_variables)

    max_duration, max_trial_num = get_max_values(config_file)
    sleep_interval = 3

    for _ in range(0, max_duration+10, sleep_interval):
        time.sleep(sleep_interval)
        status = get_experiment_status(STATUS_URL)
        if status in ['DONE', 'ERROR'] or get_failed_trial_jobs(TRIAL_JOBS_URL):
            break

    if status != 'DONE' or get_succeeded_trial_num(TRIAL_JOBS_URL) < max_trial_num:
        print_trial_job_log(training_service, TRIAL_JOBS_URL)
        raise AssertionError('Failed to finish in maxExecDuration')

def case_excluded(name, excludes):
    if name is None:
        return False
    if excludes is not None:
        excludes = excludes.split(',')
        for e in excludes:
            if name in e or e in name:
                return True
    return False

def run(args):
    it_config = get_yml_content(args.config)

    for test_case_config in it_config['testCases']:
        name = test_case_config['name']
        if case_excluded(name, args.exclude):
            print('{} excluded'.format(name))
            continue
        if args.case and name and args.case not in name:
            continue
        print('{}Testing: {}{}'.format(GREEN, name, CLEAR))
        time.sleep(5)
        begin_time = time.time()

        run_test_case(test_case_config, it_config, args)
        print(GREEN + 'Test %s: TEST PASS IN %d mins' % (name, (time.time() - begin_time)/60) + CLEAR)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--nni_source_dir", type=str, default='../')
    parser.add_argument("--case", type=str, default=None)
    parser.add_argument("--exclude", type=str, default=None)
    parser.add_argument("--ts", type=str, choices=['local', 'remote', 'pai', 'kubeflow', 'frameworkcontroller'], default='local')
    args = parser.parse_args()

    run(args)
