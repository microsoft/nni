# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Launch trial keeper daemon and print some info in JSON format.

Before running this script, you should:

  1. Create the working directory with "create_trial_keeper_dir.py".
  2. Create "launcher_config.json" in that directory.

Example of "launcher_config.json":

    {
        "experimentId": "EXP_ID",
        "environmentId": "ENV_ID",
        "platform": "remote",
        "managerCommandChannel": "ws://1.2.3.4/platform/remote0/worker1",
        "logLevel": "debug"
    }

Usage:

    python -m nni.tools.nni_manager_scripts.launch_trial_keeper TRIAL_KEEPER_DIR
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict

import psutil

import nni_node

Config = Dict[str, str]

def main() -> None:
    trial_keeper_dir = Path(sys.argv[1])

    # create trial_keeper_config.json

    config = json.loads((trial_keeper_dir / 'launcher_config.json').read_text('utf_8'))
    config['experimentsDirectory'] = str(Path.home() / 'nni-experiments')  # TODO: configurable
    config['pythonInterpreter'] = sys.executable

    config_json = json.dumps(config, ensure_ascii=False, indent=4)
    (trial_keeper_dir / 'trial_keeper_config.json').write_text(config_json)

    # prepare upload directory

    upload_dir = Path(
        config['experimentsDirectory'],
        config['experimentId'],
        'environments',
        config['environmentId'],
        'upload'
    )
    upload_dir.mkdir(parents=True, exist_ok=True)

    # launch process

    node_dir = Path(nni_node.__path__[0])  # type: ignore
    node = str(node_dir / ('node.exe' if sys.platform == 'win32' else 'node'))
    main_js = str(node_dir / 'common/trial_keeper/main.js')
    cmd = [node, '--max-old-space-size=4096', '--trace-uncaught', main_js, str(trial_keeper_dir)]

    if sys.platform == 'win32':
        success = _windows_launch(trial_keeper_dir, cmd)
    else:
        success = _posix_launch(trial_keeper_dir, cmd)

    # save and print result

    if success:
        result = {'success': True, 'uploadDirectory': str(upload_dir)}
    else:
        err = (trial_keeper_dir / 'trial_keeper.stderr').read_text('utf_8')
        result = {'success': False, 'stderr': err}
    print(json.dumps(result, ensure_ascii=False), flush=True)

def _posix_launch(trial_keeper_dir, command) -> bool:
    stdout = (trial_keeper_dir / 'trial_keeper.stdout').open('a')
    stderr = (trial_keeper_dir / 'trial_keeper.stderr').open('a')
    proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, preexec_fn=os.setpgrp)  # type: ignore
    (trial_keeper_dir / 'trial_keeper.pid').write_text(str(proc.pid))

    while True:
        if proc.poll() is not None:
            return False
        if (trial_keeper_dir / 'success.flag').exists():
            return True
        time.sleep(0.1)

def _windows_launch(trial_keeper_dir, command) -> bool:
    # Popen flags can only detach process from console, not login session
    # https://serverfault.com/questions/1044393

    script = [
        'powershell',
        'Invoke-WmiMethod',  # this is windows' nohup
        '-Class', 'Win32_Process',
        '-Name', 'Create',
        '-ArgumentList', "'" + ' '.join(command) + "'"
    ]
    proc = subprocess.run(script, capture_output=True)

    if proc.returncode != 0:
        err = f'PowerShell script error: {proc.returncode}\nstdout: {proc.stdout}\nstderr: {proc.stderr}'
        (trial_keeper_dir / 'trial_keeper.stderr').write_text(err)
        return False

    result = {}
    for line in proc.stdout.decode().split('\n'):
        if line.strip():
            k, v = line.split(':', 1)
            result[k.strip()] = v.strip()

    if int(result['ReturnValue']) != 0:
        err = f'Non-zero return value\nstdout: {proc.stdout}\nstderr: {proc.stderr}'
        (trial_keeper_dir / 'trial_keeper.stderr').write_text(err)
        return False

    pid = int(result['ProcessId'])
    while True:
        if not psutil.pid_exists(pid):
            return False
        if (trial_keeper_dir / 'success.flag').exists():
            return True
        time.sleep(0.1)

if __name__ == '__main__':
    main()
