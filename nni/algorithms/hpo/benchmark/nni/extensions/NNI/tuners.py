from amlb.benchmark import TaskConfig

import nni

from nni.utils import MetricType 

from nni.algorithms.hpo.hyperopt_tuner import HyperoptTuner
from nni.algorithms.hpo.evolution_tuner import EvolutionTuner
from nni.algorithms.hpo.smac_tuner.smac_tuner import SMACTuner
from nni.algorithms.hpo.gp_tuner.gp_tuner import GPTuner
from nni.algorithms.hpo.metis_tuner.metis_tuner import MetisTuner
from nni.algorithms.hpo.hyperband_advisor import Hyperband
from nni.algorithms.hpo.bohb_advisor.bohb_advisor import BOHB
from nni.algorithms.hpo.bohb_advisor.config_generator import CG_BOHB


def get_tuner(config: TaskConfig):
    # Users may add their customized Tuners here 
    if config.framework_params['tuner_type'] == 'tpe':
        return HyperoptTuner('tpe'), 'TPE Tuner'

    elif config.framework_params['tuner_type'] == 'random_search':
        return HyperoptTuner('random_search'), 'Random Search Tuner'

    elif config.framework_params['tuner_type'] == 'anneal':
        return HyperoptTuner('anneal'), 'Annealing Tuner'
    
    elif config.framework_params['tuner_type'] == 'evolution':
        return EvolutionTuner(), 'Evolution Tuner'

    elif config.framework_params['tuner_type'] == 'smac':
        return SMACTuner(), 'SMAC Tuner'

    elif config.framework_params['tuner_type'] == 'gp':
        return GPTuner(), 'GP Tuner'

    elif config.framework_params['tuner_type'] == 'metis':
        return MetisTuner(), 'Metis Tuner'

    elif config.framework_params['tuner_type'] == 'hyperband':
        if 'max_resource' in config.framework_params:
            tuner = Hyperband(R=config.framework_params['max_resource'])
        else:
            tuner = Hyperband()
        return tuner, 'Hyperband Advisor'
    
    elif config.framework_params['tuner_type'] == 'bohb':
        if 'max_resource' in config.framework_params:
            tuner = BOHB(max_budget=config.framework_params['max_resource'])     
        else:
            tuner = BOHB(max_budget=60)  
        return tuner, 'BOHB Advisor'
        
    else:
        raise RuntimeError('The requested tuner type in framework.yaml is unavailable.')

    
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
        if isinstance(self.core, nni.tuner.Tuner):
            self.core_type = 'tuner'
        elif isinstance(self.core, nni.runtime.msg_dispatcher_base.MsgDispatcherBase):
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

    
