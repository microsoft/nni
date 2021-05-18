import logging

from .tuners import NNITuner
from .run_experiment import *

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.results import save_predictions_to_file
from amlb.utils import Timer


log = logging.getLogger(__name__)


def validate_config(config: TaskConfig):
    if 'tuner_type' not in config.framework_params:
        raise RuntimeError('framework.yaml does not have a "tuner_type" field.')
    if 'limit_type' not in config.framework_params:
        raise RuntimeError('framework.yaml does not have a "limit_type" field.')
    if config.framework_params['limit_type'] not in ['time', 'ntrials']:
        raise RuntimeError('"limit_type" field must be "time" or "ntrials".') 
    if 'limit' not in config.framework_params:
        raise RuntimeError('framework.yaml does not have a "limit" field.')
    else:
        try:
            _ = int(config.framework_params['limit'])
        except:
            raise RuntimeError('"limit" field must be an integer.')  
                
    
def run(dataset: Dataset, config: TaskConfig):
    validate_config(config)
    tuner = NNITuner(config)
    if config.framework_params['limit_type']  == 'time':
        log.info("Tuning {} with NNI {} with a maximum time of {}s\n"
                 .format(config.framework_params['arch_type'], tuner.description, config.framework_params['limit']))
    elif config.framework_params['limit_type'] == 'ntrials':
        log.info("Tuning {} with NNI {} with a maximum number of trials of {}\n"
                 .format(config.framework_params['arch_type'], tuner.description, config.framework_params['limit']))
        log.info("Note: any time constraints are ignored.")

    probabilities, predictions, train_timer, y_test = run_experiment(dataset, config, tuner, log)
    
    save_predictions_to_file(dataset=dataset,
                             output_file=config.output_predictions_file,
                             probabilities=probabilities,
                             predictions=predictions,
                             truth=y_test)

    return dict(
        models_count=1,
        training_duration=train_timer.duration
    )
