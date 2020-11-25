# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from colorama import Fore

NNICTL_HOME_DIR = os.path.join(os.path.expanduser('~'), '.local', 'nnictl')

NNI_HOME_DIR = os.path.join(os.path.expanduser('~'), 'nni-experiments')

ERROR_INFO = 'ERROR: '
NORMAL_INFO = 'INFO: '
WARNING_INFO = 'WARNING: '

DEFAULT_REST_PORT = 8080
REST_TIME_OUT = 20

EXPERIMENT_SUCCESS_INFO = Fore.GREEN + 'Successfully started experiment!\n' + Fore.RESET + \
                          '------------------------------------------------------------------------------------\n' \
                          'The experiment id is %s\n'\
                          'The Web UI urls are: %s\n' \
                          '------------------------------------------------------------------------------------\n\n' \
                          'You can use these commands to get more information about the experiment\n' \
                          '------------------------------------------------------------------------------------\n' \
                          '         commands                       description\n' \
                          '1. nnictl experiment show        show the information of experiments\n' \
                          '2. nnictl trial ls               list all of trial jobs\n' \
                          '3. nnictl top                    monitor the status of running experiments\n' \
                          '4. nnictl log stderr             show stderr log content\n' \
                          '5. nnictl log stdout             show stdout log content\n' \
                          '6. nnictl stop                   stop an experiment\n' \
                          '7. nnictl trial kill             kill a trial job by id\n' \
                          '8. nnictl --help                 get help information about nnictl\n' \
                          '------------------------------------------------------------------------------------\n' \
                          'Command reference document https://nni.readthedocs.io/en/latest/Tutorial/Nnictl.html\n' \
                          '------------------------------------------------------------------------------------\n'

LOG_HEADER = '-----------------------------------------------------------------------\n' \
             '                Experiment start time %s\n' \
             '-----------------------------------------------------------------------\n'

EXPERIMENT_START_FAILED_INFO = 'There is an experiment running in the port %d, please stop it first or set another port!\n' \
                               'You could use \'nnictl stop --port [PORT]\' command to stop an experiment!\nOr you could ' \
                               'use \'nnictl create --config [CONFIG_PATH] --port [PORT]\' to set port!\n'

EXPERIMENT_INFORMATION_FORMAT = '----------------------------------------------------------------------------------------\n' \
                     '                Experiment information\n' \
                     '%s\n' \
                     '----------------------------------------------------------------------------------------\n'

EXPERIMENT_DETAIL_FORMAT = 'Id: %s    Name: %s    Status: %s    Port: %s    Platform: %s    StartTime: %s    EndTime: %s\n'

EXPERIMENT_MONITOR_INFO = 'Id: %s    Status: %s    Port: %s    Platform: %s    \n' \
                          'StartTime: %s    Duration: %s'

TRIAL_MONITOR_HEAD = '-------------------------------------------------------------------------------------\n' + \
                    '%-15s %-25s %-25s %-15s \n' % ('trialId', 'startTime', 'endTime', 'status') + \
                     '-------------------------------------------------------------------------------------'

TRIAL_MONITOR_CONTENT = '%-15s %-25s %-25s %-15s'

TRIAL_MONITOR_TAIL = '-------------------------------------------------------------------------------------\n\n\n'

INSTALLABLE_PACKAGE_META = {
    'SMAC': {
        'type': 'tuner',
        'class_name': 'nni.algorithms.hpo.smac_tuner.smac_tuner.SMACTuner',
        'code_sub_dir': 'smac_tuner',
        'class_args_validator': 'nni.algorithms.hpo.smac_tuner.smac_tuner.SMACClassArgsValidator'
    },
    'BOHB': {
        'type': 'advisor',
        'class_name': 'nni.algorithms.hpo.bohb_advisor.bohb_advisor.BOHB',
        'code_sub_dir': 'bohb_advisor',
        'class_args_validator': 'nni.algorithms.hpo.bohb_advisor.bohb_advisor.BOHBClassArgsValidator'
    },
    'PPOTuner': {
        'type': 'tuner',
        'class_name': 'nni.algorithms.hpo.ppo_tuner.ppo_tuner.PPOTuner',
        'code_sub_dir': 'ppo_tuner',
        'class_args_validator': 'nni.algorithms.hpo.ppo_tuner.ppo_tuner.PPOClassArgsValidator'
    }
}

TUNERS_SUPPORTING_IMPORT_DATA = {
    'TPE',
    'Anneal',
    'GridSearch',
    'MetisTuner',
    'BOHB',
    'SMAC',
    'BatchTuner'
}

TUNERS_NO_NEED_TO_IMPORT_DATA = {
    'Random',
    'Hyperband'
}

SCHEMA_TYPE_ERROR = '%s should be %s type!'
SCHEMA_RANGE_ERROR = '%s should be in range of %s!'
SCHEMA_PATH_ERROR = '%s path not exist!'
