import contextlib
import json
import os
import subprocess
import requests
import traceback
import yaml

EXPERIMENT_DONE_SIGNAL = '"Experiment done"'

def read_last_line(file_name):
    try:
        *_, last_line = open(file_name)
        return last_line.strip()
    except (FileNotFoundError, ValueError):
        return None

def remove_files(file_list):
    for file_path in file_list:
        with contextlib.suppress(FileNotFoundError):
            os.remove(file_path)

def get_yml_content(file_path):
    '''Load yaml file content'''
    with open(file_path, 'r') as file:
        return yaml.load(file)

def dump_yml_content(file_path, content):
    '''Dump yaml file content'''
    with open(file_path, 'w') as file:
        file.write(yaml.dump(content, default_flow_style=False))

def setup_experiment(installed = True):
    if not installed:
        os.environ['PATH'] = os.environ['PATH'] + ':' + os.environ['PWD']
        sdk_path = os.path.abspath('../src/sdk/pynni')
        cmd_path = os.path.abspath('../tools')
        pypath = os.environ.get('PYTHONPATH')
        if pypath:
            pypath = ':'.join([pypath, sdk_path, cmd_path])
        else:
            pypath = ':'.join([sdk_path, cmd_path])
        os.environ['PYTHONPATH'] = pypath

def fetch_experiment_config(experiment_url):
    experiment_profile = requests.get(experiment_url)
    experiment_id = json.loads(experiment_profile.text)['id']
    experiment_path = os.path.join(os.environ['HOME'], 'nni/experiments', experiment_id)
    nnimanager_log_path = os.path.join(experiment_path, 'log', 'nnimanager.log')

    return nnimanager_log_path

def check_experiment_status(nnimanager_log_path):
    assert os.path.exists(nnimanager_log_path), 'Experiment starts failed'
    cmds = ['cat', nnimanager_log_path, '|', 'grep', EXPERIMENT_DONE_SIGNAL]
    completed_process = subprocess.run(' '.join(cmds), shell = True)
    
    return completed_process.returncode == 0