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
import json
import shutil
from .constants import HOME_DIR

class Config:
    '''a util class to load and save config'''
    def __init__(self, port):
        config_path = os.path.join(HOME_DIR, str(port))
        os.makedirs(config_path, exist_ok=True)
        self.config_file = os.path.join(config_path, '.config')
        self.config = self.read_file()

    def get_all_config(self):
        '''get all of config values'''
        return json.dumps(self.config, indent=4, sort_keys=True, separators=(',', ':'))

    def set_config(self, key, value):
        '''set {key:value} paris to self.config'''
        self.config = self.read_file()
        self.config[key] = value
        self.write_file()

    def get_config(self, key):
        '''get a value according to key'''
        return self.config.get(key)

    def copy_metadata_to_new_path(self, path):
        '''copy metadata to a new path'''
        if not os.path.exists(path):
            os.mkdir(path)
        shutil.copy(self.config_file, path)

    def write_file(self):
        '''save config to local file'''
        if self.config:
            try:
                with open(self.config_file, 'w') as file:
                    json.dump(self.config, file)
            except IOError as error:
                print('Error:', error)
                return

    def read_file(self):
        '''load config from local file'''
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as file:
                    return json.load(file)
            except ValueError:
                return {}
        return {}
