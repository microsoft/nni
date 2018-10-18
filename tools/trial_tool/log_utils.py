# ============================================================================================================================== #
# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ============================================================================================================================== #

import argparse
import errno
import json
import os
import re
import requests

from .constants import BASE_URL, DEFAULT_REST_PORT, DEFAULT_HDFS_PORT
from .rest_utils import rest_get, rest_post, rest_put, rest_delete
from .url_utils import gen_update_metrics_url, gen_read_task_url
from .hdfsClientUtility import copyDirectoryToHdfs
from pyhdfs import HdfsClient

NNI_SYS_DIR = os.environ['NNI_SYS_DIR']
NNI_TRIAL_JOB_ID = os.environ['NNI_TRIAL_JOB_ID']
NNI_EXP_ID = os.environ['NNI_EXP_ID']

class LogManager():
    '''
    Copy log to hdfs, and inform TrainingService 
    '''
    def __init__(self, local_dir, hdfs_dir, hdfs_host, user_name):
        self.local_dir = local_dir
        self.hdfs_dir = hdfs_dir
        self.hdfs_host = hdfs_host
        self.user_name = user_name
        self.task_queue = []
    
    def copyData(self):
        try:
            hdfs_client = HdfsClient(hosts='{0}:{1}'.format(self.hdfs_host, DEFAULT_HDFS_PORT), user_name=self.user_name, timeout=5)
            if copyDirectoryToHdfs(self.local_dir, self.hdfs_dir, hdfs_client):
                print('copy directory from {0} to {1} success!'.format(self.local_dir, self.hdfs_dir))
                return True
            else:
                print('copy directory from {0} to {1} failed!'.format(self.local_dir, self.hdfs_dir))
                return False
        except:
            return False
    
def get_task_from_training_service(log_manager, nnimanager_ip):
    '''
    Detect if it's time to copy data to hdfs
    '''
    try:
        response = rest_post(gen_read_task_url(BASE_URL.format(nnimanager_ip), DEFAULT_REST_PORT, NNI_EXP_ID, NNI_TRIAL_JOB_ID), json.dumps(result), 10)
        result = json.loads(response.text)
        print('get task from training service')
        print(result.text)
        if result.get('task'):
            log_manager.task_queue.append(1)
    except Exception as exception:
        print(exception)
        pass