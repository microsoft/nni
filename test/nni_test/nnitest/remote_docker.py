# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import argparse
from subprocess import check_output, check_call
import socket
import random
import re

def detect_port(port):
    '''Detect if the port is used, return True if the port is used'''
    socket_test = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        socket_test.connect(('127.0.0.1', int(port)))
        socket_test.close()
        return True
    except:
        return False

def find_port():
    '''Find a port which is free'''
    port = random.randint(10000, 20000)
    while detect_port(port):
        port = random.randint(10000, 20000)
    return port

def find_wheel_package(dir):
    '''Find the wheel package uploaded to this machine'''
    regular = re.compile('^nni-.*\.whl$')
    for file_name in os.listdir(dir):
        if regular.search(file_name):
            return file_name
    return None

def start_container(image, name, nnimanager_os):
    '''Start docker container, generate a port in /tmp/nnitest/{name}/port file'''
    port = find_port()
    source_dir = '/tmp/nnitest/' + name
    run_cmds = ['docker', 'run', '-d', '-t', '-p', str(port) + ':22', '--name', name, '--mount', 'type=bind,source=' + source_dir + ',target=/tmp/nni', image]
    output = check_output(run_cmds)
    commit_id = output.decode('utf-8')
    
    if nnimanager_os == 'windows':
        wheel_name = find_wheel_package(os.path.join(source_dir, 'nni-remote/deployment/pypi/dist'))
    else:
        wheel_name = find_wheel_package(os.path.join(source_dir, 'dist'))
        
    if not wheel_name:
        print('Error: could not find wheel package in {0}'.format(source_dir))
        exit(1)
        
    def get_dist(wheel_name):
        '''get the wheel package path'''
        if nnimanager_os == 'windows':
            return '/tmp/nni/nni-remote/deployment/pypi/dist/{0}'.format(wheel_name)
        else:
            return '/tmp/nni/dist/{0}'.format(wheel_name)
        
    pip_cmds = ['docker', 'exec', name, 'python3', '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools==41.0.0']
    check_call(pip_cmds)
    sdk_cmds = ['docker', 'exec', name, 'python3', '-m', 'pip', 'install', get_dist(wheel_name)]
    check_call(sdk_cmds)
    with open(source_dir + '/port', 'w') as file:
        file.write(str(port))

def stop_container(name):
    '''Stop docker container'''
    stop_cmds = ['docker', 'container', 'stop', name]
    check_call(stop_cmds)
    rm_cmds = ['docker', 'container', 'rm', name]
    check_call(rm_cmds)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['start', 'stop'], dest='mode', help='start or stop a container')
    parser.add_argument('--name', required=True, dest='name', help='the name of container to be used')
    parser.add_argument('--image', dest='image', help='the image to be used')
    parser.add_argument('--os', dest='os', default='unix', choices=['unix', 'windows'], help='nniManager os version')
    args = parser.parse_args()
    if args.mode == 'start':
        start_container(args.image, args.name, args.os)
    else:
        stop_container(args.name)
