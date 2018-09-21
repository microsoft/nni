#!/usr/bin/env python3

import contextlib
import json
import os
import subprocess
import time
import traceback
import yaml

GREEN = '\33[32m'
RED = '\33[31m'
CLEAR = '\33[0m'

TUNER_LIST = ['TPE', 'Random', 'Anneal', 'Evolution']

def read_last_line(file_name):
    try:
        *_, last_line = open(file_name)
        return last_line.strip()
    except (FileNotFoundError, ValueError):
        return None

def get_yml_content(file_path):
    '''Load yaml file content'''
    with open(file_path, 'r') as file:
        return yaml.load(file)

def dump_yml_content(file_path, content):
    '''Dump yaml file content'''
    with open(file_path, 'w') as file:
        file.write(yaml.dump(content, default_flow_style=False))

def switch_tuner(tuner_name):
    '''Change tuner in config.yml'''
    config_path = 'local.yml'
    experiment_config = get_yml_content(config_path)
    experiment_config['tuner'] = {
        'builtinTunerName': tuner_name,
        'classArgs': {
            'optimize_mode': 'maximize'
        }
    }
    dump_yml_content(config_path, experiment_config)

def test_builtin_tuner(tuner_name):
    with contextlib.suppress(FileNotFoundError):
        os.remove('/tmp/nni_tuner_result.txt')
    switch_tuner(tuner_name)

    print('Testing %s...'%tuner_name)
    proc = subprocess.run(['nnictl', 'create', '--config', 'local.yml'])
    assert proc.returncode == 0, '`nnictl create` failed with code %d' % proc.returncode

    time.sleep(16)
    tuner_status = read_last_line('/tmp/nni_tuner_result.txt')
    
    assert tuner_status is not None, 'Failed to finish in 16 sec'
    assert tuner_status == 'DONE', 'Tuner exited with error'

if __name__ == '__main__':

    os.environ['PATH'] = os.environ['PATH'] + ':' + os.environ['PWD']
    # Test each built-in tuner
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
