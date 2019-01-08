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

import argparse
import glob
import subprocess
import sys
import time
import traceback

from utils import setup_experiment, get_experiment_status, get_yml_content, parse_max_duration_time, get_succeeded_trial_num

GREEN = '\33[32m'
RED = '\33[31m'
CLEAR = '\33[0m'

STATUS_URL = 'http://localhost:8080/api/v1/nni/check-status'
TRIAL_JOBS_URL = 'http://localhost:8080/api/v1/nni/trial-jobs'

def run_test(config_file):
    '''run test per configuration file'''

    print('Testing %s...' % config_file)
    proc = subprocess.run(['nnictl', 'create', '--config', config_file])
    assert proc.returncode == 0, '`nnictl create` failed with code %d' % proc.returncode

    max_duration, max_trial_num = get_max_values(config_file)
    #print(max_duration, max_trial_num)
    sleep_interval = 3

    for _ in range(0, max_duration, sleep_interval):
        time.sleep(sleep_interval)
        status = get_experiment_status(STATUS_URL)
        #print('experiment status:', status)
        if status == 'DONE':
            num_succeeded = get_succeeded_trial_num(TRIAL_JOBS_URL)
            assert num_succeeded == max_trial_num, 'only %d succeeded trial jobs, there should be %d' % (num_succeeded, max_trial_num)
            break

    assert status == 'DONE', 'Failed to finish in maxExecDuration'

def get_max_values(config_file):
    experiment_config = get_yml_content(config_file)
    return parse_max_duration_time(experiment_config['maxExecDuration']), experiment_config['maxTrialNum']

def run(config_files=None):
    '''test all configuration files'''
    if config_files is None:
        config_files = glob.glob('./test_config/**/*.test.yml')
    print(config_files)

    for config_file in config_files:
        try:
            # sleep 5 seconds here, to make sure previous stopped exp has enough time to exit to avoid port conflict
            time.sleep(5)
            run_test(config_file)
            print(GREEN + 'Test %s: TEST PASS' % (config_file) + CLEAR)
        except Exception as error:
            print(RED + 'Test %s: TEST FAIL' % (config_file) + CLEAR)
            print('%r' % error)
            traceback.print_exc()
            raise error
        finally:
            subprocess.run(['nnictl', 'stop'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False)
    args = parser.parse_args()

    installed = (sys.argv[-1] != '--preinstall')
    setup_experiment(installed)

    config_files = None
    if args.config:
        config_files = args.config.split(',')
    run(config_files)
