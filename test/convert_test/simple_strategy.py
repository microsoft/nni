import json
import logging
import random
import os

from nni.retiarii import Model, submit_models, wait_models
from nni.retiarii import get_base_model_ir, get_specified_mutators, get_trainer
from nni.retiarii import Sampler

_logger = logging.getLogger(__name__)

class RandomSampler(Sampler):
    def choice(self, candidates, mutator, model, index):
        return random.choice(candidates)

def simple_startegy():
    try:
        _logger.info('stargety start...')
        while True:
            model = get_base_model_ir()
            _logger.info('apply mutators...')
            applied_mutators = get_specified_mutators()
            _logger.info('mutators: {}'.format(applied_mutators))
            random_sampler = RandomSampler()
            for mutator in applied_mutators:
                _logger.info('mutate model...')
                mutator.bind_sampler(random_sampler)
                model = mutator.apply(model)
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
