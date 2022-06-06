# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from subprocess import call, check_output
import sys
import os
import time
import signal
import psutil
from .common_utils import print_error, print_warning


def check_output_command(file_path, head=None, tail=None):
    """call check_output command to read content from a file"""
    if os.path.exists(file_path):
        if sys.platform == 'win32':
            cmds = ['powershell.exe', 'type', file_path]
            if head:
                cmds += ['|', 'select', '-first', str(head)]
            elif tail:
                cmds += ['|', 'select', '-last', str(tail)]
            return check_output(cmds, shell=True).decode('utf-8')
        else:
            cmds = ['cat', file_path]
            if head:
                cmds = ['head', '-' + str(head), file_path]
            elif tail:
                cmds = ['tail', '-' + str(tail), file_path]
            return check_output(cmds, shell=False).decode('utf-8')
    else:
        print_error('{0} does not exist!'.format(file_path))
        exit(1)


def kill_command(pid, timeout=60):
    """Kill the process of pid (with a terminate signal).
    Waiting up to 60 seconds until the process is killed.
    """
    # TODO: The input argument should better be Popen rather than pid.
    if sys.platform == 'win32':
        try:
            process = psutil.Process(pid=pid)
            process.send_signal(signal.CTRL_BREAK_EVENT)
        except psutil.NoSuchProcess:
            print_warning(f'Tried to kill process (pid = {pid}), but the process does not exist.')
    else:
        cmds = ['kill', str(pid)]
        call(cmds)
    if not _wait_till_process_killed(pid, timeout):
        print_warning(
            f'One subprocess (pid = {pid}) still exists after {timeout} seconds since sending the killing signal is sent. '
            'Perhaps the shutdown of this process has hang for some reason. You might have to kill it by yourself.'
        )


def install_package_command(package_name):
    """
    Install python package from pip.

    Parameters
    ----------
    package_name: str
        The name of package to be installed.
    """
    call(_get_pip_install() + [package_name], shell=False)


def install_requirements_command(requirements_path):
    """
    Install packages from `requirements.txt` in `requirements_path`.

    Parameters
    ----------
    requirements_path: str
        Path to the directory that contains `requirements.txt`.
    """
    return call(_get_pip_install() + ["-r", requirements_path], shell=False)


def _wait_till_process_killed(pid, timeout):
    keyboard_interrupted = False
    time_count = 0
    # Usually, a process is killed very quickly.
    # This little nap will save 1 second.
    time.sleep(0.01)
    while True:
        try:
            # Implementation of waiting
            while time_count < timeout:
                pid_running = _check_pid_running(pid)
                if not pid_running:
                    return True
                time.sleep(1)
                time_count += 1
            return False
        except KeyboardInterrupt:
            # Warn at the first keyboard interrupt and do nothing
            # Stop at the second
            if keyboard_interrupted:
                print_warning('Wait of process killing cancelled.')
                # I think throwing an exception is more reasonable.
                # Another option is to return false here, which is also acceptable.
                raise
            print_warning(
                f'Waiting for the cleanup of a process (pid = {pid}). '
                'We suggest you waiting for it to complete. '
                'Press Ctrl-C again if you intend to interrupt the cleanup.'
            )
            keyboard_interrupted = True

    # Actually we will never reach here
    return False


def _check_pid_running(pid):
    # Check whether process still running.
    # FIXME: the correct impl should be using ``proc.poll()``
    # Using pid here is unsafe.
    # We should make Popen object directly accessible.
    if sys.platform == 'win32':
        # NOTE: Tests show that the behavior of psutil is unreliable, and varies from runs to runs.
        # Also, Windows didn't explicitly handle child / non-child process.
        # This might be a potential problem.
        try:
            psutil.Process(pid).wait(timeout=0)
            return False
        except psutil.TimeoutExpired:
            return True
        except psutil.NoSuchProcess:
            return False
    else:
        try:
            indicator, _ = os.waitpid(pid, os.WNOHANG)
            return indicator == 0
        except ChildProcessError:
            # One of the reasons we reach here is: pid may be not a child process.
            # In that case, we can use the famous kill 0 to poll the process.
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False


def _get_pip_install():
    python = "python" if sys.platform == "win32" else "python3"
    ret = [python, "-m", "pip", "install"]
    if "CONDA_DEFAULT_ENV" not in os.environ and "VIRTUAL_ENV" not in os.environ and \
            (sys.platform != "win32" and os.getuid() != 0):  # on unix and not running in root
        ret.append("--user")  # not in virtualenv or conda
    return ret

def call_pip_install(source):
    return call(_get_pip_install() + [source])

def call_pip_uninstall(module_name):
    python = "python" if sys.platform == "win32" else "python3"
    cmd = [python, "-m", "pip", "uninstall", module_name]
    return call(cmd)
