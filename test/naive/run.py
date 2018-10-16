#!/usr/bin/env python3

import contextlib
import json
import os
import subprocess
import requests
import sys
import time
import traceback

GREEN = '\33[32m'
RED = '\33[31m'
CLEAR = '\33[0m'

class Integration_test():
    def __init__(self):
        self.experiment_url = 'http://localhost:51188/api/v1/nni/experiment'
        self.experiment_id = None
        self.experiment_suspended_signal = '"Experiment suspended"'

    def read_last_line(self, file_name):
        try:
            *_, last_line = open(file_name)
            return last_line.strip()
        except (FileNotFoundError, ValueError):
            return None

    def fetch_experiment_config(self):
        experiment_profile = requests.get(self.experiment_url)
        self.experiment_id = json.loads(experiment_profile.text)['id']
        self.experiment_path = os.path.join(os.environ['HOME'], 'nni/experiments', self.experiment_id)
        self.nnimanager_log_path = os.path.join(self.experiment_path, 'log', 'nnimanager.log')

    def check_experiment_status(self):
        assert os.path.exists(self.nnimanager_log_path), 'Experiment starts failed'
        cmds = ['cat', self.nnimanager_log_path, '|', 'grep', self.experiment_suspended_signal]
        completed_process = subprocess.run(' '.join(cmds), shell = True)
        
        return completed_process.returncode == 0

    def remove_files(self, file_list):
        for file_path in file_list:
            with contextlib.suppress(FileNotFoundError):
                os.remove(file_path)

    def run(self, installed = True):
        if not installed:
            os.environ['PATH'] = os.environ['PATH'] + ':' + os.environ['PWD']
            sdk_path = os.path.abspath('../../src/sdk/pynni')
            cmd_path = os.path.abspath('../../tools')
            pypath = os.environ.get('PYTHONPATH')
            if pypath:
                pypath = ':'.join([pypath, sdk_path, cmd_path])
            else:
                pypath = ':'.join([sdk_path, cmd_path])
            os.environ['PYTHONPATH'] = pypath

        to_remove = ['tuner_search_space.json', 'tuner_result.txt', 'assessor_result.txt']
        self.remove_files(to_remove)

        proc = subprocess.run(['nnictl', 'create', '--config', 'local.yml'])
        assert proc.returncode == 0, '`nnictl create` failed with code %d' % proc.returncode

        print('Spawning trials...')
        time.sleep(1)
        self.fetch_experiment_config()
        current_trial = 0

        for _ in range(60):
            time.sleep(1)

            tuner_status = self.read_last_line('tuner_result.txt')
            assessor_status = self.read_last_line('assessor_result.txt')
            experiment_status = self.check_experiment_status()

            assert tuner_status != 'ERROR', 'Tuner exited with error'
            assert assessor_status != 'ERROR', 'Assessor exited with error'

            if experiment_status:
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
        ss2 = json.load(open('tuner_search_space.json'))
        assert ss1 == ss2, 'Tuner got wrong search space'

        # Waiting for naive_trial to report_final_result
        time.sleep(2)
        tuner_result = set(open('tuner_result.txt'))
        expected = set(open('expected_tuner_result.txt'))
        # Trials may complete before NNI gets assessor's result,
        # so it is possible to have more final result than expected
        assert tuner_result.issuperset(expected), 'Bad tuner result'

        assessor_result = set(open('assessor_result.txt'))
        expected = set(open('expected_assessor_result.txt'))
        assert assessor_result == expected, 'Bad assessor result'

if __name__ == '__main__':
    installed = (sys.argv[-1] != '--preinstall')

    ic = Integration_test()
    try:
        ic.run(installed)
        # TODO: check the output of rest server
        print(GREEN + 'PASS' + CLEAR)
    except Exception as error:
        print(RED + 'FAIL' + CLEAR)
        print('%r' % error)
        traceback.print_exc()
        raise error

    subprocess.run(['nnictl', 'stop', '--port', '51188'])
