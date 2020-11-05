# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from subprocess import call, check_output
import sys
import os
import signal
import psutil
from .common_utils import print_error


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


def kill_command(pid):
    """kill command"""
    if sys.platform == 'win32':
        psutil.Process(pid).terminate()
    else:
        cmds = ['kill', str(pid)]
        call(cmds)


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
