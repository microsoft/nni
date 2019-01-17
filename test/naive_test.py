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

import json
import subprocess
import sys
import time
import traceback

from utils import is_experiment_done, fetch_nni_log_path, read_last_line, remove_files, setup_experiment
from utils import GREEN, RED, CLEAR, EXPERIMENT_URL

def run():
    '''run naive integration test'''
    to_remove = ['tuner_search_space.json', 'tuner_result.txt', 'assessor_result.txt']
    to_remove = list(map(lambda file: 'naive_test/' + file, to_remove))
    remove_files(to_remove)

    proc = subprocess.run(['nnictl', 'create', '--config', 'naive_test/local.yml'])
    assert proc.returncode == 0, '`nnictl create` failed with code %d' % proc.returncode

    print('Spawning trials...')

    nnimanager_log_path = fetch_nni_log_path(EXPERIMENT_URL)
    current_trial = 0

    for _ in range(120):
        time.sleep(1)

        tuner_status = read_last_line('naive_test/tuner_result.txt')
        assessor_status = read_last_line('naive_test/assessor_result.txt')
        experiment_status = is_experiment_done(nnimanager_log_path)

        assert tuner_status != 'ERROR', 'Tuner exited with error'
        assert assessor_status != 'ERROR', 'Assessor exited with error'

        if experiment_status:
            break

        if tuner_status is not None:
            for line in open('naive_test/tuner_result.txt'):
                if line.strip() == 'ERROR':
                    break
                trial = int(line.split(' ')[0])
                if trial > current_trial:
                    current_trial = trial
                    print('Trial #%d done' % trial)

    assert experiment_status, 'Failed to finish in 2 min'

    ss1 = json.load(open('naive_test/search_space.json'))
    ss2 = json.load(open('naive_test/tuner_search_space.json'))
    assert ss1 == ss2, 'Tuner got wrong search space'

    tuner_result = set(open('naive_test/tuner_result.txt'))
    expected = set(open('naive_test/expected_tuner_result.txt'))
    # Trials may complete before NNI gets assessor's result,
    # so it is possible to have more final result than expected
    assert tuner_result.issuperset(expected), 'Bad tuner result'

    assessor_result = set(open('naive_test/assessor_result.txt'))
    expected = set(open('naive_test/expected_assessor_result.txt'))
    assert assessor_result == expected, 'Bad assessor result'

if __name__ == '__main__':
    installed = (sys.argv[-1] != '--preinstall')
    setup_experiment(installed)
    try:
        run()
        # TODO: check the output of rest server
        print(GREEN + 'PASS' + CLEAR)
    except Exception as error:
        print(RED + 'FAIL' + CLEAR)
        print('%r' % error)
        traceback.print_exc()
        sys.exit(1)
    finally:
        subprocess.run(['nnictl', 'stop'])
