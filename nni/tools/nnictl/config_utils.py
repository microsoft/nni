# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
import shutil
import sqlite3
import time
from .constants import NNICTL_HOME_DIR
from .command_utils import print_error
from .common_utils import get_file_lock
from ...experiment.config.convert import config_v0_to_v1

class Config:
    '''a util class to load and save config'''
    def __init__(self, experiment_id: str, log_dir: str):
        self.experiment_id = experiment_id
        self.conn = sqlite3.connect(log_dir)
        self.cursor = self.conn.cursor()
        self.refresh_config()

    def refresh_config(self):
        '''refresh to get latest config'''
        sql = 'select * from ExperimentProfile where id=? order by revision DESC'
        args = (self.experiment_id,)
        self.config = config_v0_to_v1(json.loads(self.cursor.execute(sql, args).fetchall()[0][0]))

    def get_config(self):
        '''get a value according to key'''
        return self.config

class Experiments:
    '''Maintain experiment list'''
    def __init__(self, home_dir=NNICTL_HOME_DIR):
        os.makedirs(home_dir, exist_ok=True)
        self.experiment_file = os.path.join(home_dir, '.experiment')
        self.lock = get_file_lock(self.experiment_file, stale=2)
        with self.lock:
            self.experiments = self.read_file()

    def add_experiment(self, expId, port, startTime, platform, experiment_name, endTime='N/A', status='INITIALIZED',
                       tag=[], pid=None, webuiUrl=[], logDir=[]):
        '''set {key:value} pairs to self.experiment'''
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
            if value is None:
                self.experiments[expId].pop(key, None)
            else:
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
                json.dump(self.experiments, file, indent=4)
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
