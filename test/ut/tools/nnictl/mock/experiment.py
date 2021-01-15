# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from pathlib import Path
from subprocess import Popen, PIPE, STDOUT
from nni.tools.nnictl.config_utils import Experiments
from nni.tools.nnictl.common_utils import print_green
from nni.tools.nnictl.command_utils import kill_command
from nni.tools.nnictl.nnictl_utils import get_yml_content

def create_mock_experiment():
    nnictl_experiment_config = Experiments()
    nnictl_experiment_config.add_experiment('xOpEwA5w', '8080', 123456,
                                            'local', 'example_sklearn-classification')
    # mock process
    cmds = ['sleep', '3600000']
    process = Popen(cmds, stdout=PIPE, stderr=STDOUT)
    nnictl_experiment_config.update_experiment('xOpEwA5w', 'pid', process.pid)
    nnictl_experiment_config.update_experiment('xOpEwA5w', 'port', 8080)
    nnictl_experiment_config.update_experiment('xOpEwA5w', 'webuiUrl', ['http://localhost:8080'])
    print_green("expriment start success, experiment id: xOpEwA5w")

def stop_mock_experiment():
    nnictl_experiment_config = Experiments()
    experiments_dict = nnictl_experiment_config.get_all_experiments()
    kill_command(experiments_dict['xOpEwA5w'].get('pid'))
    nnictl_experiment_config = Experiments()
    nnictl_experiment_config.remove_experiment('xOpEwA5w')

def generate_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('id', nargs='?')
    parser.add_argument('--port', '-p', type=int, dest='port')
    parser.add_argument('--all', '-a', action='store_true')
    parser.add_argument('--head', type=int)
    parser.add_argument('--tail', type=int)
    return parser

def generate_args():
    parser = generate_args_parser()
    args = parser.parse_args(['xOpEwA5w'])
    return args
