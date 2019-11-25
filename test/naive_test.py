# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import os.path as osp
import json
import subprocess
import sys
import time
import traceback

from utils import is_experiment_done, get_experiment_id, get_nni_log_path, read_last_line, remove_files, setup_experiment, detect_port, snooze
from utils import GREEN, RED, CLEAR, EXPERIMENT_URL

def naive_test():
    '''run naive integration test'''
    to_remove = ['tuner_search_space.json', 'tuner_result.txt', 'assessor_result.txt']
    to_remove = list(map(lambda file: osp.join('naive_test', file), to_remove))
    remove_files(to_remove)

    if sys.platform == 'win32':
        config_file = 'local_win32.yml'
    else:
        config_file = 'local.yml'
    proc = subprocess.run(['nnictl', 'create', '--config', osp.join('naive_test' , config_file)])
    assert proc.returncode == 0, '`nnictl create` failed with code %d' % proc.returncode

    print('Spawning trials...')

    nnimanager_log_path = get_nni_log_path(EXPERIMENT_URL)
    current_trial = 0

    for _ in range(120):
        time.sleep(1)

        tuner_status = read_last_line(osp.join('naive_test', 'tuner_result.txt'))
        assessor_status = read_last_line(osp.join('naive_test', 'assessor_result.txt'))
        experiment_status = is_experiment_done(nnimanager_log_path)

        assert tuner_status != 'ERROR', 'Tuner exited with error'
        assert assessor_status != 'ERROR', 'Assessor exited with error'

        if experiment_status:
            break

        if tuner_status is not None:
            for line in open(osp.join('naive_test', 'tuner_result.txt')):
                if line.strip() == 'ERROR':
                    break
                trial = int(line.split(' ')[0])
                if trial > current_trial:
                    current_trial = trial
                    print('Trial #%d done' % trial)

    assert experiment_status, 'Failed to finish in 2 min'

    ss1 = json.load(open(osp.join('naive_test', 'search_space.json')))
    ss2 = json.load(open(osp.join('naive_test', 'tuner_search_space.json')))
    assert ss1 == ss2, 'Tuner got wrong search space'

    tuner_result = set(open(osp.join('naive_test', 'tuner_result.txt')))
    expected = set(open(osp.join('naive_test', 'expected_tuner_result.txt')))
    # Trials may complete before NNI gets assessor's result,
    # so it is possible to have more final result than expected
    print('Tuner result:', tuner_result)
    print('Expected tuner result:', expected)
    assert tuner_result.issuperset(expected), 'Bad tuner result'

    assessor_result = set(open(osp.join('naive_test', 'assessor_result.txt')))
    expected = set(open(osp.join('naive_test', 'expected_assessor_result.txt')))
    assert assessor_result == expected, 'Bad assessor result'

    subprocess.run(['nnictl', 'stop'])
    snooze()

def stop_experiment_test():
    '''Test `nnictl stop` command, including `nnictl stop exp_id` and `nnictl stop all`.
    Simple `nnictl stop` is not tested here since it is used in all other test code'''
    subprocess.run(['nnictl', 'create', '--config', osp.join('tuner_test', 'local.yml'), '--port', '8080'], check=True)
    subprocess.run(['nnictl', 'create', '--config', osp.join('tuner_test', 'local.yml'), '--port', '8888'], check=True)
    subprocess.run(['nnictl', 'create', '--config', osp.join('tuner_test', 'local.yml'), '--port', '8989'], check=True)
    subprocess.run(['nnictl', 'create', '--config', osp.join('tuner_test', 'local.yml'), '--port', '8990'], check=True)

    # test cmd 'nnictl stop id`
    experiment_id = get_experiment_id(EXPERIMENT_URL)
    proc = subprocess.run(['nnictl', 'stop', experiment_id])
    assert proc.returncode == 0, '`nnictl stop %s` failed with code %d' % (experiment_id, proc.returncode)
    snooze()
    assert not detect_port(8080), '`nnictl stop %s` failed to stop experiments' % experiment_id

    # test cmd `nnictl stop --port`
    proc = subprocess.run(['nnictl', 'stop', '--port', '8990'])
    assert proc.returncode == 0, '`nnictl stop %s` failed with code %d' % (experiment_id, proc.returncode)
    snooze()
    assert not detect_port(8990), '`nnictl stop %s` failed to stop experiments' % experiment_id

    # test cmd `nnictl stop --all`
    proc = subprocess.run(['nnictl', 'stop', '--all'])
    assert proc.returncode == 0, '`nnictl stop --all` failed with code %d' % proc.returncode
    snooze()
    assert not detect_port(8888) and not detect_port(8989), '`nnictl stop --all` failed to stop experiments'


if __name__ == '__main__':
    installed = (sys.argv[-1] != '--preinstall')
    setup_experiment(installed)
    try:
        naive_test()
        stop_experiment_test()
        # TODO: check the output of rest server
        print(GREEN + 'PASS' + CLEAR)
    except Exception as error:
        print(RED + 'FAIL' + CLEAR)
        print('%r' % error)
        traceback.print_exc()
        sys.exit(1)
