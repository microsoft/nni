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

import unittest
import json
import sys
from pyhdfs import HdfsClient
sys.path.append("..")
from trial.hdfsClientUtility import copyFileToHdfs, copyDirectoryToHdfs
import os
import shutil
import random
import string

class HDFSClientUtilityTest(unittest.TestCase):
    '''Unit test for hdfsClientUtility.py'''
    def setUp(self):
        self.hdfs_file_path = '../../.vscode/hdfsInfo.json'
        self.hdfs_config = None
        try:
            with open(self.hdfs_file_path, 'r') as file:
                self.hdfs_config = json.load(file)
        except Exception as exception:
            print(exception)

        self.hdfs_client = HdfsClient(hosts='{0}:{1}'.format(self.hdfs_config['host'], '50070'), user_name=self.hdfs_config['userName'])

    def get_random_name(self, length):
        return ''.join(random.sample(string.ascii_letters + string.digits, length))

    def test_copy_file_run(self):
        '''test copyFileToHdfs'''
        file_name = self.get_random_name(8)
        file_content = 'hello world!'

        with open('./{}'.format(file_name), 'w') as file:
            file.write(file_content)

        result = copyFileToHdfs('./{}'.format(file_name), '/{0}/{1}'.format(self.hdfs_config['userName'], file_name), self.hdfs_client)
        self.assertTrue(result)

        file_list = self.hdfs_client.listdir('/{0}'.format(self.hdfs_config['userName']))
        self.assertIn(file_name, file_list)

        hdfs_file_name = self.get_random_name(8)
        self.hdfs_client.copy_to_local('/{0}/{1}'.format(self.hdfs_config['userName'], file_name), './{}'.format(hdfs_file_name))
        self.assertTrue(os.path.exists('./{}'.format(hdfs_file_name)))

        with open('./{}'.format(hdfs_file_name), 'r') as file:
            content = file.readline()
            self.assertEqual(file_content, content)
        #clean up
        os.remove('./{}'.format(file_name))
        os.remove('./{}'.format(hdfs_file_name))
        self.hdfs_client.delete('/{0}/{1}'.format(self.hdfs_config['userName'], file_name))

    def test_copy_directory_run(self):
        '''test copyDirectoryToHdfs'''
        directory_name = self.get_random_name(8)
        file_name_list = [self.get_random_name(8), self.get_random_name(8)]
        file_content = 'hello world!'

        os.makedirs('./{}'.format(directory_name))
        for file_name in file_name_list:
            with open('./{0}/{1}'.format(directory_name, file_name), 'w') as file:
                file.write(file_content)

        result = copyDirectoryToHdfs('./{}'.format(directory_name), '/{0}/{1}'.format(self.hdfs_config['userName'], directory_name), self.hdfs_client)
        self.assertTrue(result)

        directory_list = self.hdfs_client.listdir('/{0}'.format(self.hdfs_config['userName']))
        self.assertIn(directory_name, directory_list)

        sub_file_list = self.hdfs_client.listdir('/{0}/{1}'.format(self.hdfs_config['userName'], directory_name))
        for file_name in file_name_list:
            self.assertIn(file_name, sub_file_list)
            #clean up
            self.hdfs_client.delete('/{0}/{1}/{2}'.format(self.hdfs_config['userName'], directory_name, file_name))
        self.hdfs_client.delete('/{0}/{1}'.format(self.hdfs_config['userName'], directory_name))

        shutil.rmtree('./{}'.format(directory_name))

if __name__ == '__main__':
    unittest.main()