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
from pyhdfs import HdfsClient

def copyDirectoryToHdfs(localDirectory, hdfsDirectory, hdfsClient):
    '''Copy directory from local to hdfs'''
    if not os.path.exists(localDirectory):
        raise Exception('Local Directory does not exist!')
        hdfsClient.mkdirs(hdfsDirectory)
    for file in os.listdir(localDirectory):
        file_path = os.path.join(localDirectory, file)
        if os.path.isdir(file_path):
            hdfs_directory = os.path.join(hdfsDirectory, file)
            try:
                copyDirectoryToHdfs(file_path, hdfs_directory, hdfsClient)
            except Exception as exception:
                print(exception)
        else:
            hdfs_file_path = os.path.join(hdfsDirectory, file)
            try:
                copyFileToHdfs(file_path, hdfs_file_path, hdfsClient)
            except Exception as exception:
                print(exception)
    return True

def copyFileToHdfs(localFilePath, hdfsFilePath, hdfsClient, override=True):
    '''Copy a local file to hdfs directory'''
    if not os.path.exists(localFilePath):
        raise Exception('Local file Path does not exist!')
    if hdfsClient.exists(hdfsFilePath):
        if override:
            hdfsClient.delete(hdfsFilePath)
        else:
            return False
    try:
        return hdfsClient.copy_from_local(localFilePath, hdfsFilePath)
    except Exception as exception:
        print(exception)
        return False