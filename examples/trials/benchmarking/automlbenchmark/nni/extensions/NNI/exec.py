import logging

from .tuners import NNITuner
from .run_experiment import *

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.results import save_predictions_to_file
from amlb.utils import Timer


log = logging.getLogger(__name__)

    
def run(dataset: Dataset, config: TaskConfig):
    if 'tuner_type' not in config.framework_params:
        raise RuntimeError('framework.yaml does not have a "tuner_type" field.')
    
    tuner = NNITuner(config)
    log.info("Tuning {} with NNI {} with a maximum time of {}s\n"
             .format(config.framework_params['arch_type'], tuner.description, config.max_runtime_seconds))

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
