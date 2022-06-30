# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess
import argparse
import time
import shlex
import signal

def test_foreground(args):
    launch_command = 'nnictl create --config {} --foreground'.format(args.config)
    print('nnictl foreground launch command: ', launch_command, flush=True)

    proc = subprocess.Popen(shlex.split(launch_command))

    time.sleep(args.timeout)
    proc.send_signal(signal.SIGINT)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--timeout", type=int, default=45)
    args = parser.parse_args()

    test_foreground(args)
