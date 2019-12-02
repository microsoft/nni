# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import argparse
import glob
import subprocess
import time
import traceback
import json

from utils import setup_experiment, get_experiment_status, get_yml_content, dump_yml_content, \
    parse_max_duration_time, get_succeeded_trial_num, deep_update, print_failed_job_log, get_failed_trial_jobs
from utils import GREEN, RED, CLEAR, STATUS_URL, TRIAL_JOBS_URL

def gen_new_config(config_file, training_service='local'):
    '''
    Generates temporary config file for integration test, the file
    should be deleted after testing.
    '''
    config = get_yml_content(config_file)
    new_config_file = config_file + '.tmp'

    it_config = get_yml_content('training_service.yml')

    # hack for kubeflow trial config
    if training_service == 'kubeflow':
        it_config[training_service]['trial']['worker']['command'] = config['trial']['command']
        config['trial'].pop('command')
        if 'gpuNum' in config['trial']:
            config['trial'].pop('gpuNum')

    deep_update(config, it_config['all'])
    deep_update(config, it_config[training_service])

    dump_yml_content(new_config_file, config)

    return new_config_file, config

def run_test(config_file, training_service, local_gpu=False):
    '''run test per configuration file'''

    new_config_file, config = gen_new_config(config_file, training_service)
    print(json.dumps(config, sort_keys=True, indent=4))

    if training_service == 'local' and not local_gpu and config['trial']['gpuNum'] > 0:
        print('no gpu, skiping: ', config_file)
        return

    try:
        proc = subprocess.run(['nnictl', 'create', '--config', new_config_file])
        assert proc.returncode == 0, '`nnictl create` failed with code %d' % proc.returncode

        max_duration, max_trial_num = get_max_values(new_config_file)
        sleep_interval = 3

        for _ in range(0, max_duration+30, sleep_interval):
            time.sleep(sleep_interval)
            status = get_experiment_status(STATUS_URL)
            if status in ['DONE', 'ERROR'] or get_failed_trial_jobs(TRIAL_JOBS_URL):
                break

        print_failed_job_log(config['trainingServicePlatform'], TRIAL_JOBS_URL)
        if status != 'DONE' or get_succeeded_trial_num(TRIAL_JOBS_URL) < max_trial_num:
            raise AssertionError('Failed to finish in maxExecDuration')
    finally:
        if os.path.exists(new_config_file):
            os.remove(new_config_file)

def get_max_values(config_file):
    '''Get maxExecDuration and maxTrialNum of experiment'''
    experiment_config = get_yml_content(config_file)
    return parse_max_duration_time(experiment_config['maxExecDuration']), experiment_config['maxTrialNum']

def run(args):
    '''test all configuration files'''
    if args.config is None:
        config_files = glob.glob('./config_test/**/*.test.yml')
    else:
        config_files = args.config.split(',')

    if args.exclude is not None:
        exclude_paths = args.exclude.split(',')
        if exclude_paths:
            for exclude_path in exclude_paths:
                config_files = [x for x in config_files if exclude_path not in x]
    print(config_files)

    for config_file in config_files:
        try:
            # sleep 5 seconds here, to make sure previous stopped exp has enough time to exit to avoid port conflict
            time.sleep(5)
            print(GREEN + 'Testing:' + config_file + CLEAR)
            begin_time = time.time()
            run_test(config_file, args.ts, args.local_gpu)
            print(GREEN + 'Test %s: TEST PASS IN %d mins' % (config_file, (time.time() - begin_time)/60) + CLEAR)
        except Exception as error:
            print(RED + 'Test %s: TEST FAIL' % (config_file) + CLEAR)
            print('%r' % error)
            traceback.print_exc()
            raise error
        finally:
            subprocess.run(['nnictl', 'stop'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--exclude", type=str, default=None)
    parser.add_argument("--ts", type=str, choices=['local', 'remote', 'pai', 'kubeflow'], default='local')
    parser.add_argument("--local_gpu", action='store_true')
    parser.add_argument("--preinstall", action='store_true')
    args = parser.parse_args()

    setup_experiment(args.preinstall)

    run(args)
