# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
import json
import tempfile
import time
import socket
import string
import random
import glob
from colorama import Fore
import filelock
import psutil
import yaml

from .constants import ERROR_INFO, NORMAL_INFO, WARNING_INFO

def get_yml_content(file_path):
    '''Load yaml file content'''
    try:
        with open(file_path, 'r', encoding='utf_8') as file:
            return yaml.safe_load(file)
    except yaml.scanner.ScannerError as err:
        print_error('yaml file format error!')
        print_error(err)
        exit(1)
    except Exception as exception:
        print_error(exception)
        exit(1)

def get_json_content(file_path):
    '''Load json file content'''
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except TypeError as err:
        print_error('json file format error!')
        print_error(err)
        return None


def print_error(*content):
    '''Print error information to screen'''
    print(Fore.RED + ERROR_INFO + ' '.join([str(c) for c in content]) + Fore.RESET)

def print_green(*content):
    '''Print information to screen in green'''
    print(Fore.GREEN + ' '.join([str(c) for c in content]) + Fore.RESET)

def print_normal(*content):
    '''Print error information to screen'''
    print(NORMAL_INFO, *content)

def print_warning(*content):
    '''Print warning information to screen'''
    print(Fore.YELLOW + WARNING_INFO + ' '.join([str(c) for c in content]) + Fore.RESET)

def detect_process(pid):
    '''Detect if a process is alive'''
    try:
        process = psutil.Process(pid)
        return process.is_running()
    except:
        return False

def detect_port(port):
    '''Detect if the port is used'''
    socket_test = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        socket_test.connect(('127.0.0.1', int(port)))
        socket_test.close()
        return True
    except:
        return False

def get_user():
    if sys.platform == 'win32':
        return os.environ['USERNAME']
    else:
        return os.environ['USER']

def generate_temp_dir():
    '''generate a temp folder'''
    def generate_folder_name():
        return os.path.join(tempfile.gettempdir(), 'nni', ''.join(random.sample(string.ascii_letters + string.digits, 8)))
    temp_dir = generate_folder_name()
    while os.path.exists(temp_dir):
        temp_dir = generate_folder_name()
    os.makedirs(temp_dir)
    return temp_dir

class SimplePreemptiveLock(filelock.SoftFileLock):
    '''this is a lock support check lock expiration, if you do not need check expiration, you can use SoftFileLock'''
    def __init__(self, lock_file, stale=-1):
        super(__class__, self).__init__(lock_file, timeout=-1)

        # FIXME: hack
        if not hasattr(self, '_lock_file'):
            self._lock_file = self.lock_file

        self._lock_file_name = '{}.{}'.format(self._lock_file, os.getpid())
        self._stale = stale

    def _acquire(self):
        open_mode = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_TRUNC
        try:
            lock_file_names = glob.glob(self._lock_file + '.*')
            for file_name in lock_file_names:
                if os.path.exists(file_name) and (self._stale < 0 or time.time() - os.stat(file_name).st_mtime < self._stale):
                    return None
            fd = os.open(self._lock_file_name, open_mode)
        except (IOError, OSError):
            pass
        else:
            self._lock_file_fd = fd
        return None

    def _release(self):
        os.close(self._lock_file_fd)
        self._lock_file_fd = None
        try:
            os.remove(self._lock_file_name)
        except OSError:
            pass
        return None

def get_file_lock(path: string, stale=-1):
    return SimplePreemptiveLock(path + '.lock', stale=stale)
