import json
import logging
import os

from nni.retiarii import Model, submit_models, wait_models
from nni.retiarii import get_base_model_ir, get_specified_mutators, get_trainer

_logger = logging.getLogger(__name__)

def simple_startegy():
    try:
        _logger.info('stargety start...')
        model = get_base_model_ir()
        _logger.info('apply mutators...')
        applied_mutators = get_specified_mutators()
        # get and apply training approach
        _logger.info('apply training approach...')
        trainer = get_trainer()
        model.apply_trainer(trainer['modulename'], trainer['args'])
        # run models
        submit_models(model)
        wait_models(model)
        _logger.info('Strategy says:', model.metric)
    except Exception as e:
        _logger.error(logging.exception('message'))


if __name__ == '__main__':
    simple_startegy()
