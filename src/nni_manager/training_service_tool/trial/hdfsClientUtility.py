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

def copyDirectoryToHdfs(localDirectory, hdfsDirectory, hdfsClient):
    '''Copy directory from local to hdfs'''
    if not os.path.exists(localDirectory):
        raise Exception('Local Directory does not exist!')
    if not pathExists(hdfsDirectory, hdfsClient):
        hdfsClient.makedirs(hdfsDirectory)
    try:
        for file in os.listdir(localDirectory):
            file_path = os.path.join(localDirectory, file)
            if os.path.isdir(file_path):
                copyDirectoryToHdfs(file_path, hdfsDirectory, hdfsClient)
            else:
                copyFileToHdfs(localDirectory, hdfsDirectory, hdfsClient)
        return True
    except:
        return False

def copyFileToHdfs(localFilePath, hdfsFilePath, hdfsClient):
    '''Copy a local file to hdfs directory'''
    if not os.path.exists(localFilePath):
        raise Exception('Local file Path does not exist!')
    if not pathExists(hdfsFilePath, hdfsClient):
        hdfsClient.makedirs(hdfsFilePath)
    try:
        hdfsClient.upload(hdfsFilePath, localFilePath, overwrite = True)
    except:
        return False
    return True

def pathExists(hdfsPath, hdfsClient):
    '''Check if an HDFS path already exists'''
    result = hdfsClient.status(hdfsPath, strict=False)
    if result is not None:
        return True
    else:
        return False
