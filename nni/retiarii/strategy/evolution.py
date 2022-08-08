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
from .utils import dry_run_for_search_space, get_targeted_model, filter_model


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
    model_filter: Callable[[Model], bool]
        Feed the model and return a bool. This will filter the models in search space and select which to submit.
    """

    def __init__(self, optimize_mode='maximize', population_size=100, sample_size=25, cycles=20000,
                 mutation_prob=0.05, on_failure='ignore', model_filter=None):
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
        self.filter = model_filter

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
        if not filter_model(self.filter, model):
            if self.on_failure == "worst":
                model.status = ModelStatus.Failed
                self._running_models.append((config, model))
        else:
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


class RandomEvolution(RegularizedEvolution):
    """
    Algorithm for random evolution.
    """
    def __init__(self, optimize_mode='maximize', population_size=10000, on_failure='ignore', model_filter=None):
        super().__init__(optimize_mode=optimize_mode, population_size=population_size, sample_size=25, cycles=100,
                 mutation_prob=0.1, on_failure=on_failure, model_filter=model_filter)

    def random(self, search_space, fixed):
        sampled = {k: random.choice(v) for k, v in search_space.items()}
        sampled.update(fixed)
        return sampled

    def run(self, base_model, applied_mutators):
        search_space = dry_run_for_search_space(base_model, applied_mutators)
        # Run the first population regardless concurrency
        _logger.info('Initializing the first population.')
        while len(self._population) + len(self._running_models) <= self.population_size:
            # try to submit new models
            while query_available_resources() > 0:
                config = self.random(search_space, {})
                self._submit_config(config, base_model, applied_mutators)
            # collect results
            self._move_succeeded_models_to_population()
            self._remove_failed_models_from_running_list()
            time.sleep(self._polling_interval)

            if len(self._population) >= self.population_size:
                break


dict2tuple = lambda d: tuple(d[k] for k in sorted(d.keys()))

class AutoformerEvolution(BaseStrategy):
    """
    Algorithm for Autoformer evolution.
    Follows evolution search in Chen et al. "AutoFormer: Searching Transformers for Visual Recognition".
    Parameters
    ----------
    mutation_size : int
        The number of individuals that should mutate in each generation. Default: 25.
    crossover_size: int
        The number of individuals that should cross over in each generation. Default: 25.
    mutation_prob : float
        Probability that mutation happens default. Default: 0.4
    depth_mutation_prob : float
        Probability that mutation happens in depth. Default: 0.2
    """
    MAX_ITER = 100
    def __init__(self, generations=20, population_size=50, parent_size=25, 
            mutation_size=25, crossover_size=25, mutation_prob=0.4, depth_mutation_prob=0.2, 
            optimize_mode='maximize', on_failure='ignore', model_filter=None):
        assert parent_size < population_size and crossover_size + mutation_size <= population_size
        assert generations > 1
        super().__init__(optimize_mode=optimize_mode, population_size=population_size, sample_size=25, cycles=100,
                 mutation_prob=0.1, on_failure=on_failure, model_filter=model_filter)

        self.generations = generations
        self.parent_size = parent_size
        self.mutation_size = mutation_size
        self.crossover_size = crossover_size
        self.depth_mutation_prob = depth_mutation_prob
        self.mutation_prob = mutation_prob

        self._cache = set()

    def _cache_hit(self, config):
        if dict2tuple(config) in self._cache:
            return True
        else:
            self._cache.add(config)
            return False

    def random(self, search_space):
        for _ in range(self.MAX_ITER):
            config = {k: random.choice(v) for k, v in search_space.items()}
            if not self._cache_hit(config):
                break
        return config

    def mutation(self, parents, search_space):
        for _ in range(self.MAX_ITER):
            config = {k: v for k, v in random.choice(parents).item()}    # copy
            # mutate depth
            if random.random() < self.depth_mutation_prob:
                config["depth"] = random.choice(search_space["depth"])
            # mutate embed_dim
            if random.random() < self.mutation_prob:
                config["embed_dim"] = random.choice(search_space["embed_dim"])
            # mlp_ratio
            for d in range(config["depth"]):
                if random.random() < self.mutation_prob:
                    config[f'mlp_ratio_{d}'] = random.choice(search_space[f'mlp_ratio_{d}'])
                if random.random() < self.mutation_prob:
                    config[f'num_head_{d}'] = random.choice(search_space[f'num_head_{d}'])

            if not self._cache_hit(config):
                break

        return config

    def crossover(self, parents, search_space):
        p1 = random.choice(parents)
        for _ in range(self.MAX_ITER):
            p2 = random.choice(parents)
            if p1["depth"] != p2["depth"]:
                continue
            config = {k: v for k, v in p1.item()}    # copy
            for k in p1.keys():
                config[k] = random.choice([p1[k], p2[k]])
            if not self._cache_hit(config):
                break

        return config

    def topk_parents(self, parents):
        reverse = self.optimize_mode=='maximize'
        parents = sorted(parents, key = lambda p: p.y, reverse = reverse)
        return [p.x for p in parents]

    def run(self, base_model, applied_mutators):
        search_space = dry_run_for_search_space(base_model, applied_mutators)
        # Run the first population regardless concurrency
        _logger.info('Initializing the first generations.')
        while True:
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

        for _ in range(self.generations - 1):
            parents = self.topk_parents([self._population])
            population = self.mutate(parents, search_space) + self.crossover(parents, search_space)
            while len(population) < self.population_size:
                config = self.random(search_space)
                population.append(config)

            while True:
                # try to submit new models
                while query_available_resources() > 0 and len(population) > 0:
                    config = population.pop()
                    self._submit_config(config, base_model, applied_mutators)
                # collect results
                self._move_succeeded_models_to_population()
                self._remove_failed_models_from_running_list()
                time.sleep(self._polling_interval)

                if len(self._running_models) + len(population) == 0:
                    break

