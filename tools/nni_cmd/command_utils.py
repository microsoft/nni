from subprocess import call, check_output
import sys
import os
from .common_utils import  print_error, print_normal, print_warning

def check_output_command(file_path, head=None, tail=None):
    '''call check_output command to read content from a file'''
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
    '''kill command'''
    if sys.platform == 'win32':
        cmds = ['powershell.exe', 'kill', str(pid)]
    else:
        cmds = ['kill', str(pid)]
    call(cmds)
