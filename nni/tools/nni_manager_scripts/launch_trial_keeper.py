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
from subprocess import Popen
import sys
import time
from typing import Dict

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

    stdout = (trial_keeper_dir / 'trial_keeper.stdout').open('ab')
    stderr = (trial_keeper_dir / 'trial_keeper.stderr').open('ab')

    node_dir = Path(nni_node.__path__[0])  # type: ignore
    node = str(node_dir / ('node.exe' if sys.platform == 'win32' else 'node'))
    main_js = str(node_dir / 'common/trial_keeper/main.js')
    cmd = [node, '--max-old-space-size=4096', '--trace-uncaught', main_js, str(trial_keeper_dir)]

    # NOTE: cwd must be node_dir, or trial keeper will not work (because of app-module-path/cwd)
    if sys.platform == 'win32':
        from subprocess import CREATE_BREAKAWAY_FROM_JOB, DETACHED_PROCESS
        flags = CREATE_BREAKAWAY_FROM_JOB | DETACHED_PROCESS
        proc = Popen(cmd, stdout=stdout, stderr=stderr, cwd=node_dir, creationflags=flags)
    else:
        proc = Popen(cmd, stdout=stdout, stderr=stderr, cwd=node_dir, preexec_fn=os.setpgrp)  # type: ignore
    (trial_keeper_dir / 'trial_keeper.pid').write_text(str(proc.pid))

    # wait for result

    while True:
        if proc.poll() is not None:
            success = False
            break
        if (trial_keeper_dir / 'success.flag').exists():
            success = True
            break
        time.sleep(0.1)

    # save and print result

    if success:
        result = {'success': True, 'uploadDirectory': str(upload_dir), 'trialKeeperDirectory': str(trial_keeper_dir)}
    else:
        err = (trial_keeper_dir / 'trial_keeper.stderr').read_text('utf_8')
        result = {'success': False, 'stderr': err}
    print(json.dumps(result, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
