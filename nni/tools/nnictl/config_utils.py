# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
import shutil
import time
from .constants import NNICTL_HOME_DIR
from .command_utils import print_error
from .common_utils import get_file_lock

class Config:
    '''a util class to load and save config'''
    def __init__(self, file_path, home_dir=NNICTL_HOME_DIR):
        config_path = os.path.join(home_dir, str(file_path))
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
    def __init__(self, home_dir=NNICTL_HOME_DIR):
        os.makedirs(home_dir, exist_ok=True)
        self.experiment_file = os.path.join(home_dir, '.experiment')
        self.lock = get_file_lock(self.experiment_file, timeout=1, stale=2)
        with self.lock:
            self.experiments = self.read_file()

    def add_experiment(self, expId, port, startTime, platform, experiment_name, endTime='N/A', status='INITIALIZED',
                       tag=[], pid=None, webuiUrl=[], logDir=[]):
        '''set {key:value} paris to self.experiment'''
        with self.lock:
            self.experiments = self.read_file()
            self.experiments[expId] = {}
            self.experiments[expId]['id'] = expId
            self.experiments[expId]['port'] = port
            self.experiments[expId]['startTime'] = startTime
            self.experiments[expId]['endTime'] = endTime
            self.experiments[expId]['status'] = status
            self.experiments[expId]['platform'] = platform
            self.experiments[expId]['experimentName'] = experiment_name
            self.experiments[expId]['tag'] = tag
            self.experiments[expId]['pid'] = pid
            self.experiments[expId]['webuiUrl'] = webuiUrl
            self.experiments[expId]['logDir'] = logDir
            self.write_file()

    def update_experiment(self, expId, key, value):
        '''Update experiment'''
        with self.lock:
            self.experiments = self.read_file()
            if expId not in self.experiments:
                return False
            self.experiments[expId][key] = value
            self.write_file()
            return True

    def remove_experiment(self, expId):
        '''remove an experiment by id'''
        with self.lock:
            self.experiments = self.read_file()
            if expId in self.experiments:
                self.experiments.pop(expId)
                fileName = expId
                if fileName:
                    logPath = os.path.join(NNICTL_HOME_DIR, fileName)
                    try:
                        shutil.rmtree(logPath)
                    except FileNotFoundError:
                        print_error('{0} does not exist.'.format(logPath))
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
            return ''

    def read_file(self):
        '''load config from local file'''
        if os.path.exists(self.experiment_file):
            try:
                with open(self.experiment_file, 'r') as file:
                    return json.load(file)
            except ValueError:
                return {}
        return {}
