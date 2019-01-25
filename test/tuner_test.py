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

import subprocess
import sys
import time
import traceback

from utils import get_yml_content, dump_yml_content, setup_experiment, fetch_nni_log_path, is_experiment_done

GREEN = '\33[32m'
RED = '\33[31m'
CLEAR = '\33[0m'

TUNER_LIST = ['GridSearch', 'BatchTuner', 'TPE', 'Random', 'Anneal', 'Evolution']
ASSESSOR_LIST = ['Medianstop']
EXPERIMENT_URL = 'http://localhost:8080/api/v1/nni/experiment'


def switch(dispatch_type, dispatch_name):
    '''Change dispatch in config.yml'''
    config_path = 'tuner_test/local.yml'
    experiment_config = get_yml_content(config_path)
    if dispatch_name in ['GridSearch', 'BatchTuner']:
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
    dump_yml_content(config_path, experiment_config)

def test_builtin_dispatcher(dispatch_type, dispatch_name):
    '''test a dispatcher whose type is dispatch_type and name is dispatch_name'''
    switch(dispatch_type, dispatch_name)

    print('Testing %s...' % dispatch_name)
    proc = subprocess.run(['nnictl', 'create', '--config', 'tuner_test/local.yml'])
    assert proc.returncode == 0, '`nnictl create` failed with code %d' % proc.returncode

    nnimanager_log_path = fetch_nni_log_path(EXPERIMENT_URL)

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
            # sleep 5 seconds here, to make sure previous stopped exp has enough time to exit to avoid port conflict
            time.sleep(5)
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
