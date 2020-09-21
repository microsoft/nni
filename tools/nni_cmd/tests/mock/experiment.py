# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import argparse
from subprocess import Popen, call, check_call, CalledProcessError, PIPE, STDOUT
from nni_cmd.config_utils import Config, Experiments
from nni_cmd.common_utils import detect_process, print_green
from nni_cmd.command_utils import kill_command
from nni_cmd.nnictl_utils import get_yml_content
import json

def create_mock_experiment():
    nnictl_experiment_config = Experiments()
    nnictl_experiment_config.add_experiment('xOpEwA5w', '8080', '1970/01/1 01:01:01', 'aGew0x',
                                            'local', 'example_sklearn-classification')
    nni_config = Config('aGew0x')
    # mock process
    cmds = ['sleep', '3600000']
    process = Popen(cmds, stdout=PIPE, stderr=STDOUT)
    nni_config.set_config('restServerPid', process.pid)
    nni_config.set_config('experimentId', 'xOpEwA5w')
    nni_config.set_config('restServerPort', 8080)
    nni_config.set_config('webuiUrl', ['http://localhost:8080'])
    print('============current dir:', os.getcwd())
    experiment_config = get_yml_content('./tests/config_files/valid/test.yml')
    nni_config.set_config('experimentConfig', experiment_config)
    print_green("expriment start success, experiment id: xOpEwA5w")

def stop_mock_experiment():
    config = Config('config', HOME_PATH)
    kill_command(config.get_config('restServerPid'))
    nnictl_experiment_config = Experiments()
    nnictl_experiment_config.remove_experiment('xOpEwA5w')

def generate_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('id', nargs='?')
    parser.add_argument('--port', '-p', dest='port')
    parser.add_argument('--all', '-a', action='store_true')
    parser.add_argument('--head', type=int)
    parser.add_argument('--tail', type=int)
    return parser

def generate_args():
    parser = generate_args_parser()
    args = parser.parse_args(['xOpEwA5w'])
    return args
