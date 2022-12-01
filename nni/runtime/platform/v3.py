# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import os
from typing import Any

from ..command_channel.http import HttpChannel
from ..env_vars import trial_env_vars

_logger = logging.getLogger(__name__)

_channel: HttpChannel = None  # type: ignore

def _init_channel() -> None:
    global _channel
    url = os.environ['NNI_TRIAL_COMMAND_CHANNEL']
    _logger.info(f'Connect to trial command channel {url}')
    _channel = HttpChannel(url)

def get_next_parameter() -> Any:
    if _channel is None:
        _init_channel()

    req = {'type': 'request_parameter'}
    _channel.send(req)

    res = _channel.receive()
    if res is None:
        _logger.error('Command channel is closed')
        return None
    assert res['type'] == 'parameter'
    return json.loads(res['parameter'])

def send_metric(string: str) -> None:
    if _channel is None:
        _init_channel()

    command = {'type': 'metric', 'metric': string}
    _channel.send(command)

def get_experiment_id() -> str:
    return trial_env_vars.NNI_EXP_ID

def get_trial_id() -> str:
    return trial_env_vars.NNI_TRIAL_JOB_ID

def get_sequence_id() -> int:
    return int(trial_env_vars.NNI_TRIAL_SEQ_ID)
