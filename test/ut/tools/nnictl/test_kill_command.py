import argparse
import multiprocessing
import os
import subprocess
import signal
import sys
import signal
import time

import pytest

from nni.tools.nnictl.command_utils import kill_command, _check_pid_running


def process_normal():
    time.sleep(360)


def process_kill_slow(kill_time=2):
    def handler_stop_signals(signum, frame):
        print('debug proceess kill: signal received')
        time.sleep(kill_time)
        print('debug proceess kill: signal processed')
        sys.exit(0)

    signal.signal(signal.SIGINT, handler_stop_signals)
    signal.signal(signal.SIGTERM, handler_stop_signals)
    print('debug process kill: sleep')
    time.sleep(360)


def process_patiently_kill():
    process = subprocess.Popen([sys.executable, __file__, '--mode', 'kill_very_slow'])
    time.sleep(1)
    kill_command(process.pid)  # wait long enough


def test_kill_process():
    process = multiprocessing.Process(target=process_normal)
    process.start()

    start_time = time.time()
    kill_command(process.pid)
    end_time = time.time()
    assert not _check_pid_running(process.pid)
    assert end_time - start_time < 1


def test_kill_process_slow_no_patience():
    process = subprocess.Popen([sys.executable, __file__, '--mode', 'kill_slow'])
    time.sleep(1)  # wait 1 second for the process to launch and register hooks
    start_time = time.time()
    kill_command(process.pid, timeout=1)  # didn't wait long enough
    end_time = time.time()
    assert _check_pid_running(process.pid)
    assert 0.5 < end_time - start_time < 2
    time.sleep(2)   # wait 2 more seconds
    assert not _check_pid_running(process.pid)


def test_kill_process_slow_patiently():
    process = subprocess.Popen([sys.executable, __file__, '--mode', 'kill_slow'])
    time.sleep(1)  # wait 1 second for the process to launch and register hooks
    start_time = time.time()
    kill_command(process.pid, timeout=3)  # wait long enough
    end_time = time.time()
    assert 1 < end_time - start_time < 5


@pytest.mark.skipif(sys.platform == 'win32', reason='Signal is tricky on windows.')
def test_kill_process_interrupted():
    # Launch a subprocess that launches and kills another subprocess
    process = multiprocessing.Process(target=process_patiently_kill)
    process.start()
    time.sleep(3)

    os.kill(process.pid, signal.SIGINT)
    # it doesn't work
    assert process.is_alive()
    time.sleep(0.5)
    # Ctrl+C again.
    os.kill(process.pid, signal.SIGINT)
    time.sleep(0.5)
    assert not process.is_alive()
    assert process.exitcode != 0


def start_new_process_group(cmd):
    # Otherwise cmd will be killed after this process is killed
    # To mock the behavior of nni experiment launch
    if sys.platform == 'win32':
        return subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        return subprocess.Popen(cmd, preexec_fn=os.setpgrp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['kill_slow', 'kill_very_slow'])

    args = parser.parse_args()
    if args.mode == 'kill_slow':
        process_kill_slow()
    elif args.mode == 'kill_very_slow':
        process_kill_slow(15)
    else:
        # debuggings here
        pass


if __name__ == '__main__':
    main()
