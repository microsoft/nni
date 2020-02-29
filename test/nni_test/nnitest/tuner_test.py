# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import os.path as osp
import subprocess
import sys
import time
import traceback

from utils import get_yml_content, dump_yml_content, setup_experiment, get_nni_log_path, is_experiment_done
from utils import GREEN, RED, CLEAR, EXPERIMENT_URL

TUNER_LIST = ['GridSearch', 'BatchTuner', 'TPE', 'Random', 'Anneal', 'Evolution']
ASSESSOR_LIST = ['Medianstop']


def get_config_file_path():
    if sys.platform == 'win32':
        config_file = osp.join('tuner_test', 'local_win32.yml')
    else:
        config_file = osp.join('tuner_test', 'local.yml')
    return config_file

def switch(dispatch_type, dispatch_name):
    '''Change dispatch in config.yml'''
    config_path = get_config_file_path()
    experiment_config = get_yml_content(config_path)
    if dispatch_name in ['GridSearch', 'BatchTuner', 'Random']:
        experiment_config[dispatch_type.lower()] = {
            'builtin' + dispatch_type + 'Name': dispatch_name
        }
    else:
        experiment_config[dispatch_type.lower()] = {
            'builtin' + dispatch_type + 'Name': dispatch_name,
            'classArgs': {
                'optimize_mode': 'maximize'
            }
        }
    if dispatch_name == 'BatchTuner':
        experiment_config['searchSpacePath'] = 'batchtuner_search_space.json'
    else:
        experiment_config['searchSpacePath'] = 'search_space.json'
    dump_yml_content(config_path, experiment_config)

def test_builtin_dispatcher(dispatch_type, dispatch_name):
    '''test a dispatcher whose type is dispatch_type and name is dispatch_name'''
    switch(dispatch_type, dispatch_name)

    print('Testing %s...' % dispatch_name)
    proc = subprocess.run(['nnictl', 'create', '--config', get_config_file_path()])
    assert proc.returncode == 0, '`nnictl create` failed with code %d' % proc.returncode

    nnimanager_log_path = get_nni_log_path(EXPERIMENT_URL)

    for _ in range(20):
        time.sleep(3)
        # check if experiment is done
        experiment_status = is_experiment_done(nnimanager_log_path)
        if experiment_status:
            break

    assert experiment_status, 'Failed to finish in 1 min'

def run(dispatch_type):
    '''test all dispatchers whose type is dispatch_type'''
    assert dispatch_type in ['Tuner', 'Assessor'], 'Unsupported dispatcher type: %s' % (dispatch_type)
    dipsatcher_list = TUNER_LIST if dispatch_type == 'Tuner' else ASSESSOR_LIST
    for dispatcher_name in dipsatcher_list:
        try:
            # Sleep here to make sure previous stopped exp has enough time to exit to avoid port conflict
            time.sleep(6)
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
