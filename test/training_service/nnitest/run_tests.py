# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import datetime
import json
import os
import subprocess
import sys
import time

import yaml

import validators
from utils import (CLEAR, EXPERIMENT_URL, GREEN, RED, REST_ENDPOINT,
                   STATUS_URL, TRIAL_JOBS_URL, deep_update, dump_yml_content,
                   get_experiment_dir, get_experiment_id,
                   get_experiment_status, get_failed_trial_jobs,
                   get_trial_stats, get_yml_content, parse_max_duration_time,
                   print_experiment_log, print_trial_job_log,
                   wait_for_port_available)

it_variables = {}


def update_training_service_config(config, training_service, config_file_path, nni_source_dir, reuse_mode='False'):
    it_ts_config = get_yml_content(os.path.join('training_service', 'config', 'training_service.yml'))
    # hack for kubeflow trial config
    if training_service == 'kubeflow' and reuse_mode == 'False':
        it_ts_config[training_service]['trial']['worker']['command'] = config['trial']['command']
        config['trial'].pop('command')
        if 'gpuNum' in config['trial']:
            config['trial'].pop('gpuNum')
    elif training_service == 'kubeflow' and reuse_mode == 'True':
        it_ts_config = get_yml_content(os.path.join('training_service', 'config', 'training_service_v2.yml'))
        print(it_ts_config)
        it_ts_config[training_service]['trainingService']['worker']['command'] = config['trialCommand']
        it_ts_config[training_service]['trainingService']['worker']['code_directory'] = config['trialCodeDirectory']

    if training_service == 'frameworkcontroller' and reuse_mode == 'False':
        it_ts_config[training_service]['trial']['taskRoles'][0]['command'] = config['trial']['command']
        config['trial'].pop('command')
        if 'gpuNum' in config['trial']:
            config['trial'].pop('gpuNum')
    elif training_service == 'frameworkcontroller' and reuse_mode == 'True':
        it_ts_config = get_yml_content(os.path.join('training_service', 'config', 'training_service_v2.yml'))
        it_ts_config[training_service]['trainingService']['taskRoles'][0]['command'] = config['trialCommand']

    if training_service == 'adl':
        # hack for adl trial config, codeDir in adl mode refers to path in container
        containerCodeDir = config['trial']['codeDir']
        # replace metric test folders to container folder
        if config['trial']['codeDir'] == '.':
            containerCodeDir = '/' + config_file_path[:config_file_path.rfind('/')]
        elif config['trial']['codeDir'] == '../naive_trial':
            containerCodeDir = '/test/training_service/config/naive_trial'
        elif '../../../' in config['trial']['codeDir']:
            # replace example folders to container folder
            containerCodeDir = config['trial']['codeDir'].replace('../../../', '/')
        it_ts_config[training_service]['trial']['codeDir'] = containerCodeDir
        it_ts_config[training_service]['trial']['command'] = 'cd {0} && {1}'.format(containerCodeDir, config['trial']['command'])

    if training_service == 'remote':
        testcase_config = get_yml_content(nni_source_dir + config_file_path)
        sharedStorage = testcase_config.get('sharedStorage')
        if sharedStorage is None:
            it_ts_config[training_service].pop('sharedStorage')
        elif str(sharedStorage.get('storageType')).lower() == 'nfs':
            it_ts_config[training_service].get('sharedStorage').pop('storageAccountKey')
        elif str(sharedStorage.get('storageType')).lower() == 'azureblob':
            it_ts_config[training_service].get('sharedStorage').pop('nfsServer')
            it_ts_config[training_service].get('sharedStorage').pop('exportedDirectory')
        else:
            it_ts_config[training_service].pop('sharedStorage')
    
    if training_service == 'hybrid':
        it_ts_config = get_yml_content(os.path.join('training_service', 'config', 'training_service_v2.yml'))
    elif reuse_mode != 'True':
        deep_update(config, it_ts_config['all'])
    deep_update(config, it_ts_config[training_service])


def prepare_config_file(test_case_config, it_config, args):
    config_path = args.nni_source_dir + test_case_config['configFile']
    test_yml_config = get_yml_content(config_path)

    # apply test case specific config
    if test_case_config.get('config') is not None:
        deep_update(test_yml_config, test_case_config['config'])

    # hack for windows
    # We've only got windows local and remote win-to-win now.
    if sys.platform == 'win32':
        test_yml_config['trial']['command'] = test_yml_config['trial']['command'].replace('python3', 'python')

    # apply training service config
    # user's gpuNum, logCollection config is overwritten by the config in training_service.yml
    # the hack for kubeflow should be applied at last step
    update_training_service_config(test_yml_config, args.ts, test_case_config['configFile'], args.nni_source_dir, args.reuse_mode)

    # generate temporary config yml file to launch experiment
    new_config_file = config_path + '.tmp'
    dump_yml_content(new_config_file, test_yml_config)
    print(yaml.safe_dump(test_yml_config, default_flow_style=False), flush=True)

    return new_config_file


def run_test_case(test_case_config, it_config, args):
    new_config_file = prepare_config_file(test_case_config, it_config, args)
    # set configFile variable
    it_variables['$configFile'] = new_config_file

    try:
        launch_test(new_config_file, args.ts, test_case_config)
        invoke_validator(test_case_config, args.nni_source_dir, args.ts)
    finally:
        stop_command = get_command(test_case_config, 'stopCommand')
        print('Stop command:', stop_command, flush=True)
        if stop_command:
            subprocess.run(stop_command, shell=True)
        exit_command = get_command(test_case_config, 'onExitCommand')
        print('Exit command:', exit_command, flush=True)
        if exit_command:
            subprocess.run(exit_command, shell=True, check=True)
        # remove tmp config file
        if os.path.exists(new_config_file):
            os.remove(new_config_file)


def invoke_validator(test_case_config, nni_source_dir, training_service):
    validator_config = test_case_config.get('validator')
    if validator_config is None or validator_config.get('class') is None:
        return

    validator = validators.__dict__[validator_config.get('class')]()
    kwargs = validator_config.get('kwargs', {})
    print('kwargs:', kwargs)
    experiment_id = get_experiment_id(EXPERIMENT_URL)
    try:
        validator(REST_ENDPOINT, get_experiment_dir(EXPERIMENT_URL), nni_source_dir, **kwargs)
    except:
        print_experiment_log(experiment_id=experiment_id)
        print_trial_job_log(training_service, TRIAL_JOBS_URL)
        raise


def get_max_values(config_file):
    experiment_config = get_yml_content(config_file)
    if experiment_config.get('maxExecDuration'):
        return parse_max_duration_time(experiment_config['maxExecDuration']), experiment_config['maxTrialNum']
    else:
        return parse_max_duration_time(experiment_config['maxExperimentDuration']), experiment_config['maxTrialNumber']


def get_command(test_case_config, commandKey):
    command = test_case_config.get(commandKey)
    if commandKey == 'launchCommand':
        assert command is not None
    if command is None:
        return None

    # replace variables
    for k in it_variables:
        command = command.replace(k, it_variables[k])

    # hack for windows, not limited to local training service
    if sys.platform == 'win32':
        command = command.replace('python3', 'python')

    return command


def launch_test(config_file, training_service, test_case_config):
    launch_command = get_command(test_case_config, 'launchCommand')
    print('launch command: ', launch_command, flush=True)

    proc = subprocess.run(launch_command, shell=True)

    assert proc.returncode == 0, 'launch command failed with code %d' % proc.returncode

    # set experiment ID into variable
    exp_var_name = test_case_config.get('setExperimentIdtoVar')
    if exp_var_name is not None:
        assert exp_var_name.startswith('$')
        it_variables[exp_var_name] = get_experiment_id(EXPERIMENT_URL)
    print('variables:', it_variables)

    max_duration, max_trial_num = get_max_values(config_file)
    print('max_duration:', max_duration, ' max_trial_num:', max_trial_num)

    if not test_case_config.get('experimentStatusCheck'):
        return

    bg_time = time.time()
    print(str(datetime.datetime.now()), ' waiting ...', flush=True)
    experiment_id = '_latest'
    try:
        # wait restful server to be ready
        time.sleep(3)
        experiment_id = get_experiment_id(EXPERIMENT_URL)
        while True:
            waited_time = time.time() - bg_time
            if waited_time > max_duration + 10:
                print('waited: {}, max_duration: {}'.format(waited_time, max_duration))
                break
            status = get_experiment_status(STATUS_URL)
            if status in ['DONE', 'ERROR']:
                print('experiment status:', status)
                break
            num_failed = len(get_failed_trial_jobs(TRIAL_JOBS_URL))
            if num_failed > 0:
                print('failed jobs: ', num_failed)
                break
            time.sleep(1)
    except:
        print_experiment_log(experiment_id=experiment_id)
        raise
    print(str(datetime.datetime.now()), ' waiting done', flush=True)
    if get_experiment_status(STATUS_URL) == 'ERROR':
        print_experiment_log(experiment_id=experiment_id)

    trial_stats = get_trial_stats(TRIAL_JOBS_URL)
    print(json.dumps(trial_stats, indent=4), flush=True)
    if status != 'DONE' or trial_stats['SUCCEEDED'] + trial_stats['EARLY_STOPPED'] < max_trial_num:
        print_experiment_log(experiment_id=experiment_id)
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


def case_included(name, cases):
    assert cases is not None
    for case in cases.split(','):
        if case in name:
            return True
    return False


def match_platform(test_case_config):
    return sys.platform in test_case_config['platform'].split(' ')


def match_training_service(test_case_config, cur_training_service):
    case_ts = test_case_config['trainingService']
    assert case_ts is not None
    if case_ts == 'all':
        return True
    if cur_training_service in case_ts.split(' '):
        return True
    return False

def match_remoteConfig(test_case_config, nni_source_dir):
    trainingservice_config = get_yml_content(os.path.join('training_service', 'config', 'training_service.yml'))
    trainingservice_config_reuse_value = str(trainingservice_config['remote']['remoteConfig']['reuse']).lower()
    testcase_config = get_yml_content(nni_source_dir + test_case_config['configFile'])
    if testcase_config.get('remoteConfig') is not None:
        if testcase_config['remoteConfig'].get('reuse') is not None:
            return str(testcase_config['remoteConfig']['reuse']).lower() == trainingservice_config_reuse_value
    return True


def run(args):
    it_config = get_yml_content(args.config)
    test_cases = it_config['testCases']

    for test_case_id, test_case_config in enumerate(test_cases, start=1):
        name = test_case_config['name']
        print(GREEN + '=' * 80 + CLEAR)
        print('## {}Testing: {}{} ##'.format(GREEN, name, CLEAR))

        # Print progress on devops
        print(f'##vso[task.setprogress value={int(test_case_id / len(test_cases) * 100)};]{name}')

        if case_excluded(name, args.exclude):
            print('{} excluded'.format(name))
            continue
        if args.cases and not case_included(name, args.cases):
            continue

        # fill test case default config
        for k in it_config['defaultTestCaseConfig']:
            if k not in test_case_config:
                test_case_config[k] = it_config['defaultTestCaseConfig'][k]
        print(json.dumps(test_case_config, indent=4))

        if not match_platform(test_case_config):
            print('skipped {}, platform {} not match [{}]'.format(name, sys.platform, test_case_config['platform']))
            continue

        if not match_training_service(test_case_config, args.ts):
            print('skipped {}, training service {} not match [{}]'.format(
                name, args.ts, test_case_config['trainingService']))
            continue

        if args.ts == 'remote':
            if not match_remoteConfig(test_case_config, args.nni_source_dir):
                print('skipped {}, remoteConfig not match.'.format(name))
                continue

        # remote mode need more time to cleanup 
        if args.ts == 'remote' or args.ts == 'hybrid':
            wait_for_port_available(8080, 240)
            wait_for_port_available(8081, 240)  # some training services need one more port to listen metrics

        # adl mode need more time to cleanup PVC
        if args.ts == 'adl' and name == 'nnictl-resume-2':
            time.sleep(30)

        begin_time = time.time()

        run_test_case(test_case_config, it_config, args)
        print('{}Test {}: TEST PASS IN {} SECONDS{}\n\n'.format(GREEN, name, int(time.time()-begin_time), CLEAR), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--nni_source_dir", type=str, default='../')
    parser.add_argument("--cases", type=str, default=None)
    parser.add_argument("--exclude", type=str, default=None)
    parser.add_argument("--reuse_mode", type=str, default='False')
    parser.add_argument("--ts", type=str, choices=['local', 'remote', 'pai',
                                                   'kubeflow', 'frameworkcontroller', 'adl', 'aml', 'hybrid'], default='local')
    args = parser.parse_args()

    run(args)
