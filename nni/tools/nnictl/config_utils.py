# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sqlite3
import nni
from .constants import NNI_HOME_DIR
from .common_utils import get_file_lock

def config_v0_to_v1(config: dict) -> dict:
    if 'clusterMetaData' not in config:
        return config
    elif 'trainingServicePlatform' in config:
        import copy
        experiment_config = copy.deepcopy(config)
        if experiment_config['trainingServicePlatform'] == 'hybrid':
            inverse_config = {'hybridConfig': experiment_config['clusterMetaData']['hybrid_config']}
            platform_list = inverse_config['hybridConfig']['trainingServicePlatforms']
            for platform in platform_list:
                inverse_config.update(_inverse_cluster_metadata(platform, experiment_config['clusterMetaData']))
            experiment_config.update(inverse_config)
        else:
            inverse_config = _inverse_cluster_metadata(experiment_config['trainingServicePlatform'], experiment_config['clusterMetaData'])
            experiment_config.update(inverse_config)
        experiment_config.pop('clusterMetaData')
        return experiment_config
    else:
        raise RuntimeError('experiment config key `trainingServicePlatform` not found')

def _inverse_cluster_metadata(platform: str, metadata_config: list) -> dict:
    inverse_config = {}
    if platform == 'local':
        inverse_config['trial'] = {}
        for kv in metadata_config:
            if kv['key'] == 'local_config':
                inverse_config['localConfig'] = kv['value']
            elif kv['key'] == 'trial_config':
                inverse_config['trial'] = kv['value']
    elif platform == 'remote':
        for kv in metadata_config:
            if kv['key'] == 'machine_list':
                inverse_config['machineList'] = kv['value']
            elif kv['key'] == 'trial_config':
                inverse_config['trial'] = kv['value']
            elif kv['key'] == 'remote_config':
                inverse_config['remoteConfig'] = kv['value']
    elif platform == 'pai':
        for kv in metadata_config:
            if kv['key'] == 'pai_config':
                inverse_config['paiConfig'] = kv['value']
            elif kv['key'] == 'trial_config':
                inverse_config['trial'] = kv['value']
    elif platform == 'kubeflow':
        for kv in metadata_config:
            if kv['key'] == 'kubeflow_config':
                inverse_config['kubeflowConfig'] = kv['value']
            elif kv['key'] == 'trial_config':
                inverse_config['trial'] = kv['value']
    elif platform == 'frameworkcontroller':
        for kv in metadata_config:
            if kv['key'] == 'frameworkcontroller_config':
                inverse_config['frameworkcontrollerConfig'] = kv['value']
            elif kv['key'] == 'trial_config':
                inverse_config['trial'] = kv['value']
    elif platform == 'aml':
        for kv in metadata_config:
            if kv['key'] == 'aml_config':
                inverse_config['amlConfig'] = kv['value']
            elif kv['key'] == 'trial_config':
                inverse_config['trial'] = kv['value']
    elif platform == 'dlc':
        for kv in metadata_config:
            if kv['key'] == 'dlc_config':
                inverse_config['dlcConfig'] = kv['value']
            elif kv['key'] == 'trial_config':
                inverse_config['trial'] = kv['value']
    elif platform == 'adl':
        for kv in metadata_config:
            if kv['key'] == 'adl_config':
                inverse_config['adlConfig'] = kv['value']
            elif kv['key'] == 'trial_config':
                inverse_config['trial'] = kv['value']
    else:
        raise RuntimeError('training service platform {} not found'.format(platform))
    return inverse_config

class Config:
    '''a util class to load and save config'''
    def __init__(self, experiment_id: str, log_dir: str):
        self.experiment_id = experiment_id
        self.conn = sqlite3.connect(os.path.join(log_dir, experiment_id, 'db', 'nni.sqlite'))
        self.refresh_config()

    def refresh_config(self):
        '''refresh to get latest config'''
        sql = 'select params from ExperimentProfile where id=? order by revision DESC'
        args = (self.experiment_id,)
        self.config = config_v0_to_v1(nni.load(self.conn.cursor().execute(sql, args).fetchone()[0]))

    def get_config(self):
        '''get a value according to key'''
        return self.config

class Experiments:
    '''Maintain experiment list'''
    def __init__(self, home_dir=NNI_HOME_DIR):
        os.makedirs(home_dir, exist_ok=True)
        self.experiment_file = os.path.join(home_dir, '.experiment')
        self.lock = get_file_lock(self.experiment_file, stale=2)
        with self.lock:
            self.experiments = self.read_file()

    def add_experiment(self, expId, port, startTime, platform, experiment_name, endTime='N/A', status='INITIALIZED',
                       tag=[], pid=None, webuiUrl=[], logDir='', prefixUrl=None):
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
            self.experiments[expId]['logDir'] = str(logDir)
            self.experiments[expId]['prefixUrl'] = prefixUrl
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
            self.write_file()

    def get_all_experiments(self):
        '''return all of experiments'''
        return self.experiments

    def write_file(self):
        '''save config to local file'''
        try:
            with open(self.experiment_file, 'w', encoding='utf_8') as file:
                nni.dump(self.experiments, file, indent=4)
        except IOError as error:
            print('Error:', error)
            return ''

    def read_file(self):
        '''load config from local file'''
        if os.path.exists(self.experiment_file):
            try:
                with open(self.experiment_file, 'r', encoding='utf_8') as file:
                    return nni.load(fp=file)
            except ValueError:
                return {}
        return {}
