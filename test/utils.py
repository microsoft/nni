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

import contextlib
import json
import os
import subprocess
import requests
import yaml

EXPERIMENT_DONE_SIGNAL = '"Experiment done"'

def read_last_line(file_name):
    '''read last line of a file and return None if file not found'''
    try:
        *_, last_line = open(file_name)
        return last_line.strip()
    except (FileNotFoundError, ValueError):
        return None

def remove_files(file_list):
    '''remove a list of files'''
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

def setup_experiment(installed=True):
    '''setup the experiment if nni is not installed'''
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

def fetch_nni_log_path(experiment_url):
    '''get nni's log path from nni's experiment url'''
    experiment_profile = requests.get(experiment_url)
    experiment_id = json.loads(experiment_profile.text)['id']
    experiment_path = os.path.join(os.environ['HOME'], 'nni/experiments', experiment_id)
    nnimanager_log_path = os.path.join(experiment_path, 'log', 'nnimanager.log')

    return nnimanager_log_path

def check_experiment_status(nnimanager_log_path):
    '''check if the experiment is done successfully'''
    assert os.path.exists(nnimanager_log_path), 'Experiment starts failed'
    cmds = ['cat', nnimanager_log_path, '|', 'grep', EXPERIMENT_DONE_SIGNAL]
    completed_process = subprocess.run(' '.join(cmds), shell=True)

    return completed_process.returncode == 0
