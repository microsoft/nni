# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import os.path as osp
import subprocess
import time
import traceback
import json
import requests

from utils import get_experiment_status, get_yml_content, parse_max_duration_time, get_succeeded_trial_num, print_failed_job_log
from utils import GREEN, RED, CLEAR, STATUS_URL, TRIAL_JOBS_URL, METRICS_URL

def run_test():
    '''run metrics test'''
    if sys.platform == 'win32':
        config_file = osp.join('metrics_test', 'metrics_win32.test.yml')
    else:
        config_file = osp.join('metrics_test', 'metrics.test.yml')

    print('Testing %s...' % config_file)
    proc = subprocess.run(['nnictl', 'create', '--config', config_file])
    assert proc.returncode == 0, '`nnictl create` failed with code %d' % proc.returncode

    max_duration, max_trial_num = get_max_values(config_file)
    sleep_interval = 3

    for _ in range(0, max_duration, sleep_interval):
        time.sleep(sleep_interval)
        status = get_experiment_status(STATUS_URL)
        #print('experiment status:', status)
        if status == 'DONE':
            num_succeeded = get_succeeded_trial_num(TRIAL_JOBS_URL)
            print_failed_job_log('local', TRIAL_JOBS_URL)
            if sys.platform == "win32":
                time.sleep(sleep_interval)  # Windows seems to have some issues on updating in time
            assert num_succeeded == max_trial_num, 'only %d succeeded trial jobs, there should be %d' % (num_succeeded, max_trial_num)
            check_metrics()
            break

    assert status == 'DONE', 'Failed to finish in maxExecDuration'

def check_metrics():
    with open(osp.join('metrics_test', 'expected_metrics.json'), 'r') as f:
        expected_metrics = json.load(f)
    print(expected_metrics)
    metrics = requests.get(METRICS_URL).json()
    intermediate_result, final_result = get_metric_results(metrics)
    assert len(final_result) == 1, 'there should be 1 final result'
    assert final_result[0] == expected_metrics['final_result']
    assert set(intermediate_result) == set(expected_metrics['intermediate_result'])

def get_metric_results(metrics):
    intermediate_result = []
    final_result = []
    for metric in metrics:
        if metric['type'] == 'PERIODICAL':
            intermediate_result.append(metric['data'])
        elif metric['type'] == 'FINAL':
            final_result.append(metric['data'])
    print(intermediate_result, final_result)

    return [round(float(x),6) for x in intermediate_result], [round(float(x), 6) for x in final_result]

def get_max_values(config_file):
    experiment_config = get_yml_content(config_file)
    return parse_max_duration_time(experiment_config['maxExecDuration']), experiment_config['maxTrialNum']

if __name__ == '__main__':
    try:
        # sleep 5 seconds here, to make sure previous stopped exp has enough time to exit to avoid port conflict
        time.sleep(5)
        run_test()
        print(GREEN + 'TEST PASS' + CLEAR)
    except Exception as error:
        print(RED + 'TEST FAIL' + CLEAR)
        print('%r' % error)
        traceback.print_exc()
        raise error
    finally:
        subprocess.run(['nnictl', 'stop'])
