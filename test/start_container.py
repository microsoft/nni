import os
import argparse
from subprocess import call
import socket
import random

CONFIG_ROOT_DIR = '~/.local/nniTest'
os.makedirs(CONFIG_ROOT_DIR)

def detect_port(port):
    '''Detect if the port is used, return True if the port is used'''
    socket_test = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    try:
        socket_test.connect(('127.0.0.1', int(port)))
        socket_test.close()
        return True
    except:
        return False

def find_port():
    '''Find a port which is free'''
    port = random.randint(5000,10000)
    while detect_port(port):
        port = random.randint(5000,10000)
    return port

if __name__ == '__main__':


