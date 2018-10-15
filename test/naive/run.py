#!/usr/bin/env python3

import contextlib
import json
import os
import subprocess
import time
import traceback

GREEN = '\33[32m'
RED = '\33[31m'
CLEAR = '\33[0m'

def read_last_line(file_name):
    try:
        *_, last_line = open(file_name)
        return last_line.strip()
    except (FileNotFoundError, ValueError):
        return None

def run():
    os.environ['PATH'] = os.environ['PATH'] + ':' + os.environ['PWD']

    with contextlib.suppress(FileNotFoundError):
        os.remove('tuner_search_space.txt')
    with contextlib.suppress(FileNotFoundError):
        os.remove('tuner_result.txt')
    with contextlib.suppress(FileNotFoundError):
        os.remove('/tmp/nni_assessor_result.txt')

    proc = subprocess.run(['nnictl', 'create', '--config', 'local.yml'])
    assert proc.returncode == 0, '`nnictl create` failed with code %d' % proc.returncode

    print('Spawning trials...')
    current_trial = 0

    for _ in range(60):
        time.sleep(1)

        tuner_status = read_last_line('/tmp/nni_tuner_result.txt')
        assessor_status = read_last_line('/tmp/nni_assessor_result.txt')

        assert tuner_status != 'ERROR', 'Tuner exited with error'
        assert assessor_status != 'ERROR', 'Assessor exited with error'

        if tuner_status == 'DONE' and assessor_status == 'DONE':
            break

        if tuner_status is not None:
            for line in open('/tmp/nni_tuner_result.txt'):
                if line.strip() in ('DONE', 'ERROR'):
                    break
                trial = int(line.split(' ')[0])
                if trial > current_trial:
                    current_trial = trial
                    print('Trial #%d done' % trial)
    subprocess.run(['nnictl', 'log', 'stderr'])
    assert tuner_status == 'DONE' and assessor_status == 'DONE', 'Failed to finish in 1 min'

    ss1 = json.load(open('search_space.json'))
    ss2 = json.load(open('/tmp/nni_tuner_search_space.json'))
    assert ss1 == ss2, 'Tuner got wrong search space'

    tuner_result = set(open('/tmp/nni_tuner_result.txt'))
    expected = set(open('expected_tuner_result.txt'))
    # Trials may complete before NNI gets assessor's result,
    # so it is possible to have more final result than expected
    assert tuner_result.issuperset(expected), 'Bad tuner result'

    assessor_result = set(open('/tmp/nni_assessor_result.txt'))
    expected = set(open('expected_assessor_result.txt'))
    assert assessor_result == expected, 'Bad assessor result'

if __name__ == '__main__':
    try:
        run()
        # TODO: check the output of rest server
        print(GREEN + 'PASS' + CLEAR)
    except Exception as error:
        print(RED + 'FAIL' + CLEAR)
        print('%r' % error)
        traceback.print_exc()
        raise error

    subprocess.run(['nnictl', 'stop'])
