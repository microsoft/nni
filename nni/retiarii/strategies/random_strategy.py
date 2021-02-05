import logging
import random
import time

from .. import Sampler, submit_models, query_available_resources
from .strategy import BaseStrategy

_logger = logging.getLogger(__name__)

class RandomSampler(Sampler):
    def choice(self, candidates, mutator, model, index):
        return random.choice(candidates)

class RandomStrategy(BaseStrategy):
    def __init__(self):
        self.random_sampler = RandomSampler()

    def run(self, base_model, applied_mutators):
        _logger.info('stargety start...')
        while True:
            avail_resource = query_available_resources()
            if avail_resource > 0:
                model = base_model
                _logger.info('apply mutators...')
                _logger.info('mutators: %s', str(applied_mutators))
                for mutator in applied_mutators:
                    mutator.bind_sampler(self.random_sampler)
                    model = mutator.apply(model)
                # run models
                submit_models(model)
            else:
                time.sleep(2)
