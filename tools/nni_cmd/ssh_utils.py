# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from .common_utils import print_error
from .command_utils import install_package_command

def check_environment():
    '''check if paramiko is installed'''
    try:
        import paramiko
    except:
        install_package_command('paramiko')
        import paramiko
    return paramiko

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

def create_ssh_sftp_client(host_ip, port, username, password, ssh_key_path, passphrase):
    '''create ssh client'''
    try:
        paramiko = check_environment()
        conn = paramiko.Transport(host_ip, port)
        if ssh_key_path is not None:
            ssh_key = paramiko.RSAKey.from_private_key_file(ssh_key_path, password=passphrase)
            conn.connect(username=username, pkey=ssh_key)
        else:
            conn.connect(username=username, password=password)
        sftp = paramiko.SFTPClient.from_transport(conn)
        return sftp
    except Exception as exception:
        print_error('Create ssh client error %s\n' % exception)

def remove_remote_directory(sftp, directory):
    '''remove a directory in remote machine'''
    try:
        files = sftp.listdir(directory)
        for file in files:
            filepath = '/'.join([directory, file])
            try:
                sftp.remove(filepath)
            except IOError:
                remove_remote_directory(sftp, filepath)
        sftp.rmdir(directory)
    except IOError as err:
        print_error(err)
