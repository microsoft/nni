import logging
import random
import time

from .. import Sampler, submit_models, query_available_resources
from .strategy import BaseStrategy

_logger = logging.getLogger(__name__)


class MetricAgnosticStrategy(BaseStrategy):
    """
    This strategy submits models once there are resources available, and does not collect metrics after submission.
    """
    def __init__(self, sampler):
        self.sampler = sampler

    def run(self, base_model, applied_mutators):
        _logger.info('Random strategy has been started.')
        for mutator in applied_mutators:
            mutator.bind_sampler(self.sampler)
        while True:
            avail_resource = query_available_resources()
            if avail_resource > 0:
                model = base_model
                _logger.info('New model created. Applied mutators: %s', str(applied_mutators))
                for mutator in applied_mutators:
                    model = mutator.apply(model)
                submit_models(model)
            else:
                time.sleep(2)


class RandomSampler(Sampler):
    def choice(self, candidates, mutator, model, index):
        return random.choice(candidates)


class RandomStrategy(MetricAgnosticStrategy):
    def __init__(self):
        super().__init__(RandomSampler())


class GridSearchSampler(Sampler):
    def choice(self, candidates, mutator, model, index):
        for mutator in applied_mutators:
            recorded_candidates, new_model = mutator.dry_run(new_model)
            sample_space.extend(recorded_candidates)
        self.tpe_sampler.update_sample_space(sample_space)