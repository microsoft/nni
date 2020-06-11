# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import posixpath
from .log_utils import LogType, nni_log

def copyHdfsDirectoryToLocal(hdfsDirectory, localDirectory, hdfsClient):
    '''Copy directory from HDFS to local'''
    if not os.path.exists(localDirectory):
        os.makedirs(localDirectory)
    try:
        listing = hdfsClient.list_status(hdfsDirectory)
    except Exception as exception:
        nni_log(LogType.Error, 'List hdfs directory {0} error: {1}'.format(hdfsDirectory, str(exception)))
        raise exception

    for f in listing:
        if f.type == 'DIRECTORY':
            subHdfsDirectory = posixpath.join(hdfsDirectory, f.pathSuffix)
            subLocalDirectory = os.path.join(localDirectory, f.pathSuffix)
            copyHdfsDirectoryToLocal(subHdfsDirectory, subLocalDirectory, hdfsClient)
        elif f.type == 'FILE':
            hdfsFilePath = posixpath.join(hdfsDirectory, f.pathSuffix)
            localFilePath = os.path.join(localDirectory, f.pathSuffix)
            copyHdfsFileToLocal(hdfsFilePath, localFilePath, hdfsClient)
        else:
            raise AssertionError('unexpected type {}'.format(f.type))

def copyHdfsFileToLocal(hdfsFilePath, localFilePath, hdfsClient, override=True):
    '''Copy file from HDFS to local'''
    if not hdfsClient.exists(hdfsFilePath):
        raise Exception('HDFS file {} does not exist!'.format(hdfsFilePath))
    try:
        file_status = hdfsClient.get_file_status(hdfsFilePath)
        if file_status.type != 'FILE':
            raise Exception('HDFS file path {} is not a file'.format(hdfsFilePath))
    except Exception as exception:
        nni_log(LogType.Error, 'Get hdfs file {0} status error: {1}'.format(hdfsFilePath, str(exception)))
        raise exception

    if os.path.exists(localFilePath) and override:
        os.remove(localFilePath)
    try:
        hdfsClient.copy_to_local(hdfsFilePath, localFilePath)
    except Exception as exception:
        nni_log(LogType.Error, 'Copy hdfs file {0} to {1} error: {2}'.format(hdfsFilePath, localFilePath, str(exception)))
        raise exception
    nni_log(LogType.Info, 'Successfully copied hdfs file {0} to {1}, {2} bytes'.format(hdfsFilePath, localFilePath, file_status.length))

def copyDirectoryToHdfs(localDirectory, hdfsDirectory, hdfsClient):
    '''Copy directory from local to HDFS'''
    if not os.path.exists(localDirectory):
        raise Exception('Local Directory does not exist!')
    hdfsClient.mkdirs(hdfsDirectory)
    result = True
    for file in os.listdir(localDirectory):
        file_path = os.path.join(localDirectory, file)
        if os.path.isdir(file_path):
            hdfs_directory = os.path.join(hdfsDirectory, file)
            try:
                result = result and copyDirectoryToHdfs(file_path, hdfs_directory, hdfsClient)
            except Exception as exception:
                nni_log(LogType.Error,
                        'Copy local directory {0} to hdfs directory {1} error: {2}'.format(file_path, hdfs_directory, str(exception)))
                result = False
        else:
            hdfs_file_path = os.path.join(hdfsDirectory, file)
            try:
                result = result and copyFileToHdfs(file_path, hdfs_file_path, hdfsClient)
            except Exception as exception:
                nni_log(LogType.Error, 'Copy local file {0} to hdfs {1} error: {2}'.format(file_path, hdfs_file_path, str(exception)))
                result = False
    return result

def copyFileToHdfs(localFilePath, hdfsFilePath, hdfsClient, override=True):
    '''Copy a local file to HDFS directory'''
    if not os.path.exists(localFilePath):
        raise Exception('Local file Path does not exist!')
    if os.path.isdir(localFilePath):
        raise Exception('localFile should not a directory!')
    if hdfsClient.exists(hdfsFilePath):
        if override:
            hdfsClient.delete(hdfsFilePath)
        else:
            return False
    try:
        hdfsClient.copy_from_local(localFilePath, hdfsFilePath)
        return True
    except Exception as exception:
        nni_log(LogType.Error, 'Copy local file {0} to hdfs file {1} error: {2}'.format(localFilePath, hdfsFilePath, str(exception)))
        return False
