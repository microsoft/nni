#!/usr/bin/env python3

import subprocess
import sys
import time
import traceback

from utils import *

GREEN = '\33[32m'
RED = '\33[31m'
CLEAR = '\33[0m'

TUNER_LIST = ['TPE', 'Random', 'Anneal', 'Evolution']
EXPERIMENT_URL = 'http://localhost:8080/api/v1/nni/experiment'


def switch_tuner(tuner_name):
    '''Change tuner in config.yml'''
    config_path = 'sdk_tuner_test/local.yml'
    experiment_config = get_yml_content(config_path)
    experiment_config['tuner'] = {
        'builtinTunerName': tuner_name,
        'classArgs': {
            'optimize_mode': 'maximize'
        }
    }
    dump_yml_content(config_path, experiment_config)

def test_builtin_tuner(tuner_name):
    remove_files(['sdk_tuner_test/nni_tuner_result.txt'])
    switch_tuner(tuner_name)

    print('Testing %s...'%tuner_name)
    proc = subprocess.run(['nnictl', 'create', '--config', 'sdk_tuner_test/local.yml'])
    assert proc.returncode == 0, '`nnictl create` failed with code %d' % proc.returncode

    nnimanager_log_path = fetch_experiment_config(EXPERIMENT_URL)
    
    for _ in range(10):
        time.sleep(3)

        # check if tuner exists with error
        tuner_status = read_last_line('tuner_result.txt')
        assert tuner_status != 'ERROR', 'Tuner exited with error'

        # check if experiment is done
        experiment_status = check_experiment_status(nnimanager_log_path)
        if experiment_status:
            break
    
    assert experiment_status, 'Failed to finish in 30 sec'

def run():
    to_remove = ['tuner_search_space.json', 'tuner_result.txt', 'assessor_result.txt']
    remove_files(to_remove)

    for tuner_name in TUNER_LIST:
        try:
            test_builtin_tuner(tuner_name)
            print(GREEN + 'Test ' +tuner_name+ ' tuner: TEST PASS' + CLEAR)
        except Exception as error:
            print(GREEN + 'Test ' +tuner_name+ ' tuner: TEST FAIL' + CLEAR)
            print('%r' % error)
            traceback.print_exc()
            raise error
        finally:
            subprocess.run(['nnictl', 'stop'])


if __name__ == '__main__':
    installed = (sys.argv[-1] != '--preinstall')
    setup_experiment(installed)

    run()
