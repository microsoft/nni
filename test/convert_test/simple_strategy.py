import json
import logging
import random
import os

from nni.retiarii import Model, submit_models, wait_models
from nni.retiarii.strategy import BaseStrategy
from nni.retiarii import Sampler


_logger = logging.getLogger(__name__)

class RandomSampler(Sampler):
    def choice(self, candidates, mutator, model, index):
        return random.choice(candidates)

class SimpleStrategy(BaseStrategy):
    def __init__(self):
        self.name = ''

    def run(self, base_model, applied_mutators, trainer):
        try:
            _logger.info('stargety start...')
            while True:
                model = base_model
                _logger.info('apply mutators...')
                _logger.info('mutators: {}'.format(applied_mutators))
                random_sampler = RandomSampler()
                for mutator in applied_mutators:
                    _logger.info('mutate model...')
                    mutator.bind_sampler(random_sampler)
                    model = mutator.apply(model)
                # get and apply training approach
                _logger.info('apply training approach...')
                model.apply_trainer(trainer['modulename'], trainer['args'])
                # run models
                submit_models(model)
                wait_models(model)
                _logger.info('Strategy says:', model.metric)
        except Exception as e:
            _logger.error(logging.exception('message'))
