# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os

from ..command_channel.http import HttpCommandChannel
from ..env_vars import trial_env_vars

_channel = HttpCommandChannel(os.environ['NNI_TRIAL_COMMAND_CHANNEL'])

def get_next_parameter():
    req = {'type': 'request_parameter'}
    _channel.send(req)

    res = _channel.receive()
    assert res['type'] == 'parameter'
    return json.loads(res['parameter'])

def send_metric(string):
    command = {'type': 'metric', 'metric': string}
    _channel.send(command)

def get_experiment_id():
    return trial_env_vars.NNI_EXP_ID

def get_trial_id():
    return trial_env_vars.NNI_TRIAL_JOB_ID

def get_sequence_id():
    return int(trial_env_vars.NNI_TRIAL_SEQ_ID)
