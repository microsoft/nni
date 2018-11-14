#!/usr/bin/env python3

import subprocess
import sys
import time
import traceback

from utils import *

GREEN = '\33[32m'
RED = '\33[31m'
CLEAR = '\33[0m'

TUNER_LIST = ['BatchTuner', 'TPE', 'Random', 'Anneal', 'Evolution']
ASSESSOR_LIST = ['Medianstop']
EXPERIMENT_URL = 'http://localhost:8080/api/v1/nni/experiment'


def switch(dispatch_type, dispatch_name):
    '''Change dispatch in config.yml'''
    config_path = 'sdk_test/local.yml'
    experiment_config = get_yml_content(config_path)
    experiment_config[dispatch_type.lower()] = {
        'builtin' + dispatch_type + 'Name': dispatch_name,
        'classArgs': {
            'optimize_mode': 'maximize'
        }
    }
    dump_yml_content(config_path, experiment_config)

def test_builtin_dispatcher(dispatch_type, dispatch_name):
    switch(dispatch_type, dispatch_name)

    print('Testing %s...' % dispatch_name)
    proc = subprocess.run(['nnictl', 'create', '--config', 'sdk_test/local.yml'])
    assert proc.returncode == 0, '`nnictl create` failed with code %d' % proc.returncode

    nnimanager_log_path = fetch_experiment_config(EXPERIMENT_URL)
    
    for _ in range(10):
        time.sleep(3)
        # check if experiment is done
        experiment_status = check_experiment_status(nnimanager_log_path)
        if experiment_status:
            break
    
    assert experiment_status, 'Failed to finish in 30 sec'

def run(dispatch_type):
    LIST = TUNER_LIST if dispatch_type == 'Tuner' else ASSESSOR_LIST
    for dispatcher_name in LIST:
        try:
            test_builtin_dispatcher(dispatch_type, dispatcher_name)
            print(GREEN + 'Test %s %s: TEST PASS' % (dispatcher_name, dispatch_type) + CLEAR)
        except Exception as error:
            print(RED + 'Test %s %s: TEST FAIL' % (dispatcher_name, dispatch_type) + CLEAR)
            print('%r' % error)
            traceback.print_exc()
            raise error
        finally:
            subprocess.run(['nnictl', 'stop'])


if __name__ == '__main__':
    installed = (sys.argv[-1] != '--preinstall')
    setup_experiment(installed)

    run('Tuner')
    run('Assessor')
