# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from pathlib import Path
import sys

import psutil

def main() -> None:
    try:
        pid_file = Path(sys.argv[1], 'trial_keeper.pid')
        pid = int(pid_file.read_text())
    except Exception:
        _exit_with_result({'alive': False, 'reason': f'Cannot read pid file {pid_file}'})

    try:
        proc = psutil.Process(pid)
    except Exception:
        _exit_with_result({'alive': False, 'reason': f'Process {pid} not found'})

    if 'nni' in ' '.join(proc.cmdline()):
        _exit_with_result({'alive': True})
    else:
        _exit_with_result({'alive': False, 'reason': f'Process {pid} is not nni'})

def _exit_with_result(result: dict) -> None:
    print(json.dumps(result), flush=True)
    sys.exit()

if __name__ == '__main__':
    main()
