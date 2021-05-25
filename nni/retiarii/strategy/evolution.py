# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import collections
import dataclasses
import logging
import random
import time

from ..execution import query_available_resources, submit_models
from ..graph import ModelStatus
from .base import BaseStrategy
from .utils import dry_run_for_search_space, get_targeted_model


_logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Individual:
    """
    A class that represents an individual.
    Holds two attributes, where ``x`` is the model and ``y`` is the metric (e.g., accuracy).
    """
    x: dict
    y: float


class RegularizedEvolution(BaseStrategy):
    """
    Algorithm for regularized evolution (i.e. aging evolution).
    Follows "Algorithm 1" in Real et al. "Regularized Evolution for Image Classifier Architecture Search".

    Parameters
    ----------
    optimize_mode : str
        Can be one of "maximize" and "minimize". Default: maximize.
    population_size : int
        The number of individuals to keep in the population. Default: 100.
    cycles : int
        The number of cycles (trials) the algorithm should run for. Default: 20000.
    sample_size : int
        The number of individuals that should participate in each tournament. Default: 25.
    mutation_prob : float
        Probability that mutation happens in each dim. Default: 0.05
    on_failure : str
        Can be one of "ignore" and "worst". If "ignore", simply give up the model and find a new one.
        If "worst", mark the model as -inf (if maximize, inf if minimize), so that the algorithm "learns" to avoid such model.
        Default: ignore.
    """

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
        samples = [p for p in self._population]  # copy population
        random.shuffle(samples)
        samples = list(samples)[:self.sample_size]
        if self.optimize_mode == 'maximize':
            parent = max(samples, key=lambda sample: sample.y)
        else:
            parent = min(samples, key=lambda sample: sample.y)
        return parent.x

    def run(self, base_model, applied_mutators):
        search_space = dry_run_for_search_space(base_model, applied_mutators)
        # Run the first population regardless concurrency
        _logger.info('Initializing the first population.')
        while len(self._population) + len(self._running_models) <= self.population_size:
            # try to submit new models
            while len(self._population) + len(self._running_models) < self.population_size:
                config = self.random(search_space)
                self._submit_config(config, base_model, applied_mutators)
            # collect results
            self._move_succeeded_models_to_population()
            self._remove_failed_models_from_running_list()
            time.sleep(self._polling_interval)

            if len(self._population) >= self.population_size:
                break

        # Resource-aware mutation of models
        _logger.info('Running mutations.')
        while self._success_count + len(self._running_models) <= self.cycles:
            # try to submit new models
            while query_available_resources() > 0 and self._success_count + len(self._running_models) < self.cycles:
                config = self.mutate(self.best_parent(), search_space)
                self._submit_config(config, base_model, applied_mutators)
            # collect results
            self._move_succeeded_models_to_population()
            self._remove_failed_models_from_running_list()
            time.sleep(self._polling_interval)

            if self._success_count >= self.cycles:
                break

    def _submit_config(self, config, base_model, mutators):
        _logger.debug('Model submitted to running queue: %s', config)
        model = get_targeted_model(base_model, mutators, config)
        submit_models(model)
        self._running_models.append((config, model))
        return model

    def _move_succeeded_models_to_population(self):
        completed_indices = []
        for i, (config, model) in enumerate(self._running_models):
            metric = None
            if self.on_failure == 'worst' and model.status == ModelStatus.Failed:
                metric = self._worst
            elif model.status == ModelStatus.Trained:
                metric = model.metric
            if metric is not None:
                individual = Individual(config, metric)
                _logger.debug('Individual created: %s', str(individual))
                self._population.append(individual)
                if len(self._population) > self.population_size:
                    self._population.popleft()
                completed_indices.append(i)
        for i in completed_indices[::-1]:
            # delete from end to start so that the index number will not be affected.
            self._success_count += 1
            self._running_models.pop(i)

    def _remove_failed_models_from_running_list(self):
        # This is only done when on_failure policy is set to "ignore".
        # Otherwise, failed models will be treated as inf when processed.
        if self.on_failure == 'ignore':
            number_of_failed_models = len([g for g in self._running_models if g[1].status == ModelStatus.Failed])
            self._running_models = [g for g in self._running_models if g[1].status != ModelStatus.Failed]
            if number_of_failed_models > 0:
                _logger.info('%d failed models are ignored. Will retry.', number_of_failed_models)
