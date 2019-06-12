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
from .constants import NNICTL_HOME_DIR

class Config:
    '''a util class to load and save config'''
    def __init__(self, file_path):
        config_path = os.path.join(NNICTL_HOME_DIR, str(file_path))
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

class Experiments:
    '''Maintain experiment list'''
    def __init__(self):
        os.makedirs(NNICTL_HOME_DIR, exist_ok=True)
        self.experiment_file = os.path.join(NNICTL_HOME_DIR, '.experiment')
        self.experiments = self.read_file()

    def add_experiment(self, id, port, time, file_name, platform):
        '''set {key:value} paris to self.experiment'''
        self.experiments[id] = {}
        self.experiments[id]['port'] = port
        self.experiments[id]['startTime'] = time
        self.experiments[id]['endTime'] = 'N/A'
        self.experiments[id]['status'] = 'INITIALIZED'
        self.experiments[id]['fileName'] = file_name
        self.experiments[id]['platform'] = platform
        self.write_file()
    
    def update_experiment(self, id, key, value):
        '''Update experiment'''
        if id not in self.experiments:
            return False
        self.experiments[id][key] = value
        self.write_file()
        return True
    
    def remove_experiment(self, id):
        '''remove an experiment by id'''
        if id in self.experiments:
            self.experiments.pop(id)
        self.write_file()
        
    def get_all_experiments(self):
        '''return all of experiments'''
        return self.experiments
    
    def write_file(self):
        '''save config to local file'''
        try:
            with open(self.experiment_file, 'w') as file:
                json.dump(self.experiments, file)
        except IOError as error:
            print('Error:', error)
            return

    def read_file(self):
        '''load config from local file'''
        if os.path.exists(self.experiment_file):
            try:
                with open(self.experiment_file, 'r') as file:
                    return json.load(file)
            except ValueError:
                return {}
        return {} 
