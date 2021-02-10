import collections
import dataclasses
import logging
import random
import time

from ..execution import submit_models
from ..graph import ModelStatus
from .base import BaseStrategy
from .utils import dry_run_for_search_space, get_targeted_model


_logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Individual:
    x: dict
    y: float


class Evolution(BaseStrategy):
    def __init__(self, optimize_mode='maximize', population_size=100, sample_size=25, cycles=20000,
                 mutation_prob=0.05, on_failure='ignore'):
        assert optimize_mode in ['maximize', 'minimize']
        assert on_failure in ['ignore', 'worst']
        assert sample_size < population_size
        self.optimize_mode = optimize_mode
        self.population_size = population_size
        self.sample_size = sample_size
        self.cycles = cycles
        self.mutation_prob = mutation_prob
        self.on_failure = on_failure

        self._worst = float('-inf') if self.optimize_mode == 'maximize' else float('inf')

        self._success_count = 0
        self._population = collections.deque()
        self._running_models = []
        self._polling_interval = 2.

    def random(self, search_space):
        return {k: random.choice(v) for k, v in search_space.items()}

    def mutate(self, parent, search_space):
        child = {}
        for k, v in parent.items():
            if random.uniform(0, 1) < self.mutation_prob:
                # NOTE: we do not exclude the original choice here for simplicity,
                # which is slightly different from the original paper.
                child[k] = random.choice(search_space[k])
            else:
                child[k] = v
        return child

    def best_parent(self):
        samples = [p for p in self.population]  # copy population
        random.shuffle(samples)
        samples = list(samples)[:self.sample_size]
        if self.optimize_mode == 'maximize':
            parent = max(samples, key=lambda sample: sample.y)
        else:
            parent = min(samples, key=lambda sample: sample.y)
        return parent.x

    def run(self, base_model, applied_mutators):
        search_space = dry_run_for_search_space(base_model, applied_mutators)
        # Run the first population regardless of the resources
        _logger.info('Initializing the first population.')
        while len(self._population) + len(self._running_models) <= self._population:
            # try to submit new models
            if len(self._population) + len(self._running_models) < self._population:
                config = self.random(search_space)
                self._submit_config(config, base_model, applied_mutators)
            # collect results
            self._remove_failed_models_from_running_list()
            self._move_succeeded_models_to_population()
            time.sleep(self._polling_interval)

        # Resource-aware mutation of models
        _logger.info('Running mutations.')
        while self._success_count + len(self._running_models) <= self.cycles:
            # try to submit new models
            if self._success_count + len(self._running_models) < self.cycles:
                config = self.mutate(self.best_parent(), search_space)
                self._submit_config(config, base_model, applied_mutators)
            # collect results
            self._remove_failed_models_from_running_list()
            self._move_succeeded_models_to_population()
            time.sleep(self._polling_interval)

    def _submit_config(self, config, base_model, mutators):
        model = get_targeted_model(base_model, mutators, config)
        submit_models(model)
        self._running_models.append((config, model))
        return model

    def _remove_failed_models_from_running_list(self):
        if self.on_failure == 'ignore':
            number_of_failed_models = len([g for g in self._running_models if g.status == ModelStatus.Failed])
            self._running_models = [g for g in self._running_models if g.status != ModelStatus.Failed]
            _logger.info('%d failed models are ignored. Will retry.', number_of_failed_models)

    def _move_succeeded_models_to_population(self):
        completed_indices = []
        for i, (config, model) in enumerate(self._running_models):
            metric = None
            if self.on_failure == 'worst' and model.status == ModelStatus.Failed:
                metric = self._worst
            elif model.status == ModelStatus.Trained:
                metric = model.metric
            if metric is not None:
                self._population.append(Individual(config, metric))
                if len(self._population) >= self.population_size:
                    self._population.popleft()
                completed_indices.append(i)
        for i in completed_indices:
            self._success_count += 1
            self._running_models.pop(i)
