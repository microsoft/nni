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
import posixpath
from pyhdfs import HdfsClient
from .command_utils import print_error

def copyHdfsDirectoryToLocal(hdfsDirectory, localDirectory, hdfsClient):
    '''Copy directory from HDFS to local'''
    if not os.path.exists(localDirectory):
        os.makedirs(localDirectory)
    try:
        listing = hdfsClient.list_status(hdfsDirectory)
    except Exception as exception:
        print_error(exception)
        exit(1)

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
            print_error('unexpected type {}'.format(f.type))
            exit(1)

def copyHdfsFileToLocal(hdfsFilePath, localFilePath, hdfsClient, override=True):
    '''Copy file from HDFS to local'''
    if not hdfsClient.exists(hdfsFilePath):
        print_error('HDFS file {} does not exist!'.format(hdfsFilePath))
    try: 
        file_status = hdfsClient.get_file_status(hdfsFilePath)
        if file_status.type != 'FILE':
            print_error('HDFS file path {} is not a file'.format(hdfsFilePath))
            exit(1)
    except Exception as exception:
        print_error(exception)
        exit(1)
    if os.path.exists(localFilePath) and override:
        os.remove(localFilePath)
    try:
        hdfsClient.copy_to_local(hdfsFilePath, localFilePath)
    except IsADirectoryError as error:
        print_error('local path could not be a directory!')
        exit(1)
    except Exception as exception:
        print_error(exception)
        exit(1)
