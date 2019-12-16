# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
import json
import socket
from pathlib import Path
import ruamel.yaml as yaml
import psutil
from .constants import ERROR_INFO, NORMAL_INFO, WARNING_INFO, COLOR_RED_FORMAT, COLOR_YELLOW_FORMAT

def get_yml_content(file_path):
    '''Load yaml file content'''
    try:
        with open(file_path, 'r') as file:
            return yaml.load(file, Loader=yaml.Loader)
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

def print_error(content):
    '''Print error information to screen'''
    print(COLOR_RED_FORMAT % (ERROR_INFO % content))

def print_normal(content):
    '''Print error information to screen'''
    print(NORMAL_INFO % content)

def print_warning(content):
    '''Print warning information to screen'''
    print(COLOR_YELLOW_FORMAT % (WARNING_INFO % content))

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

def get_python_dir(sitepackages_path):
    if sys.platform == "win32":
        return str(Path(sitepackages_path))
    else:
        return str(Path(sitepackages_path).parents[2])
