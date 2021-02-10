import collections
import dataclasses
import logging
import random

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

        self._succeed_count = 0
        self._population = collections.deque()
        self._running_models = []

    def random(self, search_space):
        return {k: random.choice(v) for k, v in search_space.items()}

    def mutate(self, config, search_space):
        new_config = {}
        for k, v in config.items():
            if random.uniform(0, 1) < self.mutation_prob:
                # NOTE: we do not exclude the original choice here for simplicity,
                # which is slightly different from the original paper.
                new_config[k] = random.choice(search_space[k])
            else:
                new_config[k] = v
        return new_config

    def run(self, base_model, applied_mutators):
        search_space = dry_run_for_search_space(base_model, applied_mutators)
        # Run the first population regardless of the resources
        _logger.info('Initializing the first population.')
        while len(self._population) + len(self._running_models) <= self._population:
            # try to submit new models
            if len(self._population) + len(self._running_models) < self._population:
                random_config = self.random(search_space)
                random_model = get_targeted_model(base_model, applied_mutators, random_config)
                submit_models(random_model)
                self._running_models.append((random_config, random_model))
            # collect results
            self._remove_failed_models_from_running_list()
            self._move_succeeded_models_to_population()

    def _is_better(self, a, b):
        if self.optimize_mode == 'maximize':
            return a > b
        else:
            return a < b

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
            self._running_models.pop(i)


class RegularizedEvolution:
    def __init__(self, search_space,
                 concurrency, population_size, sample_size, cycles, mutation_prob,
                 reward_fn, command, setup):
        self.search_space = search_space
        self.concurrency = concurrency
        self.population_size = population_size
        self.command = command
        self.setup = setup
        self.population_size = population_size
        self.sample_size = sample_size
        self.cycles = cycles
        self.mutation_prob = mutation_prob
        self.reward_fn = reward_fn
        assert self.cycles >= self.population_size >= self.sample_size

        self.population = collections.deque()

    def train_and_eval(self, config):
        pid = get_trial_manager().submit_new_trial(self.command, config, self.setup)

        while True:
            try:
                metrics = get_trial_manager().query_metrics(pid)
                if metrics is not None:
                    break
                time.sleep(5)
                continue
            except TrialFailed:
                _logger.warning(f'Config: {config}. Trial failed and use -inf as metrics.')
                metrics = float('-inf')
                break
        return self.reward_fn(config, metrics)

    def random_config(self):
        config = {}
        for k, v in SearchSpaceUtils.flatten_search_space(self.search_space).items():
            config[k] = v.random()
        _logger.info(f'Generated random config: {config}')
        return SearchSpaceUtils.restore_config(config, self.search_space)

    def mutate_config(self, parent_config):
        parent_config = SearchSpaceUtils.flatten_config(parent_config)
        config = {}
        for k, v in SearchSpaceUtils.flatten_search_space(self.search_space).items():
            config[k] = parent_config[k]
            if random.uniform(0, 1) < self.mutation_prob:
                config[k] = v.random(excludes=[parent_config[k]])
        _logger.info(f'Generated mutated config: {config}')
        return SearchSpaceUtils.restore_config(config, self.search_space)

    def import_(self, individuals):
        self.individuals = sorted(individuals, key=lambda i: i.reward)[-self.population_size:]
        random.shuffle(self.individuals)
        _logger.info(f'Imported individuals: {self.individuals}')

    def _run_random(self):
        individual = Individual(self.random_config(), None)
        individual.reward = self.train_and_eval(individual.config)
        self.population.append(individual)

    def _run_mutation(self):
        # Sample randomly chosen models from the current population.
        try:
            _lock.acquire()
            samples = copy.deepcopy(self.population)
        finally:
            _lock.release()
        random.shuffle(samples)
        samples = list(samples)[:self.population_size]
        parent = max(samples, key=lambda i: i.reward)

        individual = Individual(self.mutate_config(parent.config), None)
        individual.reward = self.train_and_eval(individual.config)
        try:
            _lock.acquire()
            self.population.append(individual)
            self.population.popleft()
        finally:
            _lock.release()

    def _wait_for_futures_and_shutdown(self, futures, pool):
        for i in futures:
            try:
                i.result()
            except:
                traceback.print_exc()
                for k in futures:
                    k.cancel()
                pool.shutdown(wait=True)
                raise
        pool.shutdown()

    def run(self):
        # Initialize the population with random models.
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency)
        fs = [pool.submit(self._run_random) for _ in range(self.population_size - len(self.population))]
        self._wait_for_futures_and_shutdown(fs, pool)

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency)
        fs = [pool.submit(self._run_mutation) for _ in range(self.cycles - self.population_size)]
        self._wait_for_futures_and_shutdown(fs, pool)
