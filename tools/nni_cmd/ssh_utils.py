# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge,
# to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
from .common_utils import print_error
from subprocess import call
from .command_utils import install_package_command

def check_environment():
    '''check if paramiko is installed'''
    try:
        import paramiko
    except:
        install_package_command('paramiko')

def copy_remote_directory_to_local(sftp, remote_path, local_path):
    '''copy remote directory to local machine'''
    try:
        os.makedirs(local_path, exist_ok=True)
        files = sftp.listdir(remote_path)
        for file in files:
            remote_full_path = os.path.join(remote_path, file)
            local_full_path = os.path.join(local_path, file)
            try:
                if sftp.listdir(remote_full_path):
                    copy_remote_directory_to_local(sftp, remote_full_path, local_full_path)
            except:
                sftp.get(remote_full_path, local_full_path)
    except Exception:
        pass

def create_ssh_sftp_client(host_ip, port, username, password):
    '''create ssh client'''
    try:
        check_environment()
        import paramiko
        conn = paramiko.Transport(host_ip, port)
        conn.connect(username=username, password=password)
        sftp = paramiko.SFTPClient.from_transport(conn)
        return sftp
    except Exception as exception:
        print_error('Create ssh client error %s\n' % exception)
