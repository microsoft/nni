import os
import argparse
from subprocess import Popen, call, check_call, CalledProcessError, PIPE, STDOUT
from nni_cmd.config_utils import Config, Experiments
from nni_cmd.common_utils import detect_process, print_green
from nni_cmd.command_utils import kill_command
from nni_cmd.nnictl_utils import get_yml_content
import json

MOCK_HOME_PATH = "./mock/nnictl_metadata"
HOME_PATH = os.path.join(os.path.expanduser('~'), '.local', 'nnictl')
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
    nni_config.set_config('webuiUrl', ['http://localhost:8080'])
    experiment_config = get_yml_content('../config_files/valid/test.yml')
    nni_config.set_config('experimentConfig', experiment_config)
    print_green("expriment start success, experiment id: xOpEwA5w")

def stop_mock_experiment():
    config = Config('config', HOME_PATH)
    kill_command(config.get_config('restServerPid'))

def generate_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('id', nargs='?')
    args = parser.parse_args(['xOpEwA5w'])
    return args
