#!/usr/bin/env python3

import contextlib
import json
import os
import subprocess
import requests
import sys
import time
import traceback
import yaml

GREEN = '\33[32m'
RED = '\33[31m'
CLEAR = '\33[0m'

TUNER_LIST = ['TPE', 'Random', 'Anneal', 'Evolution']

class Builtin_tuner_test():
    def __init__(self):
        self.experiment_url = 'http://localhost:8080/api/v1/nni/experiment'
        self.experiment_id = None
        self.experiment_done_signal = '"Experiment done"'
        self.to_remove = ['tuner_search_space.json', 'tuner_result.txt', 'assessor_result.txt']

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
        cmds = ['cat', self.nnimanager_log_path, '|', 'grep', self.experiment_done_signal]
        completed_process = subprocess.run(' '.join(cmds), shell = True)
        
        return completed_process.returncode == 0

    def remove_files(self, file_list):
        for file_path in file_list:
            with contextlib.suppress(FileNotFoundError):
                os.remove(file_path)

    def get_yml_content(self, file_path):
        '''Load yaml file content'''
        with open(file_path, 'r') as file:
            return yaml.load(file)

    def dump_yml_content(self, file_path, content):
        '''Dump yaml file content'''
        with open(file_path, 'w') as file:
            file.write(yaml.dump(content, default_flow_style=False))

    def switch_tuner(self, tuner_name):
        '''Change tuner in config.yml'''
        config_path = 'local.yml'
        experiment_config = self.get_yml_content(config_path)
        experiment_config['tuner'] = {
            'builtinTunerName': tuner_name,
            'classArgs': {
                'optimize_mode': 'maximize'
            }
        }
        self.dump_yml_content(config_path, experiment_config)

    def test_builtin_tuner(self, tuner_name):
        self.remove_files(['nni_tuner_result.txt'])
        self.switch_tuner(tuner_name)

        print('Testing %s...'%tuner_name)
        proc = subprocess.run(['nnictl', 'create', '--config', 'local.yml'])
        assert proc.returncode == 0, '`nnictl create` failed with code %d' % proc.returncode

        time.sleep(1)
        self.fetch_experiment_config()
        
        for _ in range(10):
            time.sleep(3)

            # check if tuner exists with error
            tuner_status = self.read_last_line('tuner_result.txt')
            assert tuner_status != 'ERROR', 'Tuner exited with error'

            # check if experiment is done
            experiment_status = self.check_experiment_status()
            if experiment_status:
                break
        
        assert experiment_status, 'Failed to finish in 30 sec'

    def run(self, installed = True):
        if not installed:
            os.environ['PATH'] = os.environ['PATH'] + ':' + os.path.join(os.environ['PWD'], '..')
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

        for tuner_name in TUNER_LIST:
            try:
                self.test_builtin_tuner(tuner_name)
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
    btt = Builtin_tuner_test()
    btt.run()
