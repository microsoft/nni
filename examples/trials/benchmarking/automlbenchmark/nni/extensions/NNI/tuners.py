# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import yaml
import importlib

import nni
from nni.runtime.config import get_config_file 
from nni.utils import MetricType 
from nni.tuner import Tuner
from nni.runtime.msg_dispatcher_base import MsgDispatcherBase

from amlb.benchmark import TaskConfig


def get_tuner_class_dict():
    config_file = str(get_config_file('registered_algorithms.yml'))
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    ret = {}
    for t in ['tuners', 'advisors']:
        for entry in config[t]:
            ret[entry['builtinName']] = entry['className']
    return ret


def get_tuner(config: TaskConfig):
    name2tuner = get_tuner_class_dict()
    if config.framework_params['tuner_type'] not in name2tuner:
        raise RuntimeError('The requested tuner type is unavailable.')
    else:
        module_name = name2tuner[config.framework_params['tuner_type']]
        tuner_name = module_name.split('.')[-1]
        module_name = '.'.join(module_name.split('.')[:-1])
        tuner_type = getattr(importlib.import_module(module_name), tuner_name)

        # special handlings for tuner initialization
        tuner = None
        if config.framework_params['tuner_type'] == 'TPE':
            tuner = tuner_type('tpe')

        elif config.framework_params['tuner_type'] == 'Random':
            tuner = tuner_type('random_search')

        elif config.framework_params['tuner_type'] == 'Anneal':
            tuner = tuner_type('anneal')

        elif config.framework_params['tuner_type'] == 'Hyperband':
            if 'max_resource' in config.framework_params:
                tuner = tuner_type(R=config.framework_params['max_resource'])
            else:
                tuner = tuner_type()

        elif config.framework_params['tuner_type'] == 'BOHB':
            if 'max_resource' in config.framework_params:
                tuner = tuner_type(max_budget=config.framework_params['max_resource'])
            else:
                tuner = tuner_type(max_budget=60)

        else:
            tuner = tuner_type()

        assert(tuner is not None)

        return tuner, config.framework_params['tuner_type']

    
class NNITuner:
    '''
    A specialized wrapper for the automlbenchmark framework.
    Abstracts the different behaviors of tuners and advisors into a tuner API. 
    '''
    def __init__(self, config: TaskConfig):
        self.config = config
        self.core, self.description = get_tuner(config)

        # 'tuner' or 'advisor'
        self.core_type = None      
        if isinstance(self.core, Tuner):
            self.core_type = 'tuner'
        elif isinstance(self.core, MsgDispatcherBase):
            self.core_type = 'advisor'
        else:
            raise RuntimeError('Unsupported tuner or advisor type') 

        # note: tuners and advisors use this variable differently
        self.cur_param_id = 0

        
    def __del__(self):
        self.handle_terminate()

        
    def update_search_space(self, search_space):
        if self.core_type == 'tuner':
            self.core.update_search_space(search_space)
            
        elif self.core_type == 'advisor':
            self.core.handle_update_search_space(search_space)
            # special initializations for BOHB Advisor
            from nni.algorithms.hpo.hyperband_advisor import Hyperband
            if isinstance(self.core, Hyperband):
                pass
            else:
                from nni.algorithms.hpo.bohb_advisor.bohb_advisor import BOHB
                from nni.algorithms.hpo.bohb_advisor.config_generator import CG_BOHB   
                if isinstance(self.core, BOHB):
                    self.core.cg = CG_BOHB(configspace=self.core.search_space,
                                           min_points_in_model=self.core.min_points_in_model,
                                           top_n_percent=self.core.top_n_percent,
                                           num_samples=self.core.num_samples,
                                           random_fraction=self.core.random_fraction,
                                           bandwidth_factor=self.core.bandwidth_factor,
                                           min_bandwidth=self.core.min_bandwidth)
                    self.core.generate_new_bracket()
                
        
    def generate_parameters(self):
        self.cur_param_id += 1
        if self.core_type == 'tuner':
            self.cur_param = self.core.generate_parameters(self.cur_param_id-1)
            return self.cur_param_id-1, self.cur_param
            
        elif self.core_type == 'advisor':
            self.cur_param = self.core._get_one_trial_job()
            hyperparams = self.cur_param['parameters'].copy()
            #hyperparams.pop('TRIAL_BUDGET')
            return self.cur_param['parameter_id'], hyperparams

        
    def receive_trial_result(self, parameter_id, parameters, value):
        if self.core_type == 'tuner':
            return self.core.receive_trial_result(parameter_id, parameters, value)

        elif self.core_type == 'advisor':
            metric_report = {}
            metric_report['parameter_id'] = parameter_id
            metric_report['trial_job_id'] = self.cur_param_id
            metric_report['type'] = MetricType.FINAL
            metric_report['value'] = str(value)
            metric_report['sequence'] = self.cur_param_id
            return self.core.handle_report_metric_data(metric_report)   

        
    def handle_terminate(self):
        if self.core_type == 'tuner':
            pass
        
        elif self.core_type == 'advisor':   
            self.core.stopping = True 

    
