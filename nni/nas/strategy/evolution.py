# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import collections
import dataclasses
import logging
import warnings
import copy
from typing import Deque

from numpy.random import RandomState

from nni.mutable import Sample, MutableAnnotation
from nni.nas.execution import ExecutionEngine
from nni.nas.space import ExecutableModelSpace
from nni.typehint import TrialMetric
from nni.nas.execution.event import ModelEvent, ModelEventType

from .base import Strategy
from .utils import DeduplicationHelper, RetrySamplingHelper


_logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Individual:
    """
    A class that represents an individual.
    Holds two attributes, where ``x`` is the model and ``y`` is the metric (e.g., accuracy).
    """
    x: Sample
    y: TrialMetric


class RegularizedEvolution(Strategy):
    """
    Algorithm for regularized evolution (i.e. aging evolution).
    Follows "Algorithm 1" in Real et al. "Regularized Evolution for Image Classifier Architecture Search",
    with several enhancements.

    Sample in this algorithm are called *individuals*.
    Specifically, the first ``population_size`` individuals are randomly sampled from the search space,
    and the rest are generated via a selection and mutation process.
    While new individuals are added to the population, the oldest one is removed to keep the population size constant.

    Parameters
    ----------
    population_size
        The number of individuals to keep in the population.
    sample_size
        The number of individuals that should participate in each tournament.
        When mutate, ``sample_size`` individuals can randomly selected from the population,
        and the best one among them will be treated as the parent.
    mutation_prob
        Probability that mutation happens in each dim.
    crossover
        If ``True``, the new individual will be a crossover between winners of two individual tournament.
        That means, two sets of ``sample_size`` individuals will be randomly selected from the population,
        and the best one in each set will be used as parents.
        Every dimension will be randomly selected from one of the parents.
    dedup
        Enforce one sample to never appear twice.
        The population might be smaller than ``population_size`` if this is set to ``True`` and the search space is small.
    seed
        Random seed.
    """

    def __init__(self, *,
                 population_size: int = 100,
                 sample_size: int = 25,
                 mutation_prob: float = 0.05,
                 crossover: bool = False,
                 dedup: bool = True,
                 seed: int | None = None,
                 **kwargs):
        super().__init__()

        if 'on_failure' in kwargs or 'cycles' in kwargs or 'model_filter' in kwargs or \
                'optimize_mode' in kwargs or 'dedup_retries' in kwargs:
            warnings.warn('on_failure, cycles, mode_filter, optimize_mode, dedup_retries are deprecated '
                          'and will be removed in the future. Specifying them has no effect now.', DeprecationWarning)
            raise NotImplementedError('on_failure != "ignore" or cycles is not None is not supported yet.')

        if not 1 <= sample_size <= population_size:
            raise ValueError('sample_size must be smaller than population_size and greater than 0.')

        self.population_size = population_size
        self.sample_size = sample_size
        self.mutation_prob = mutation_prob
        self.crossover = crossover

        self._dedup_helper = DeduplicationHelper(raise_on_dup=True) if dedup else None
        self._retry_helper = RetrySamplingHelper()

        # Current population. All finished running.
        self._population: Deque[Individual] = collections.deque()
        # Models that are still running. Not in the population.
        self._running_models: list[ExecutableModelSpace] = []

        self._random_state = RandomState(seed)

        self._individual_counter = 0

    def extra_repr(self) -> str:
        return f'population_size={self.population_size}, sample_size={self.sample_size}, ' + \
            f'mutation_prob={self.mutation_prob}, crossover={self.crossover}, ' + \
            f'dedup={self._dedup_helper is not None}'

    def random(self) -> ExecutableModelSpace:
        """Get a new sample via random sampling."""
        sample: Sample = {}
        model = self.model_space.random(memo=sample, random_state=self._random_state)
        if self._dedup_helper is not None:
            self._dedup_helper.dedup(sample)
        self._individual_counter += 1
        _logger.info('[Individual %4d] Random: %s', self._individual_counter, sample)
        return model

    def new_individual(self) -> ExecutableModelSpace:
        """Get a new sample via mutation from the parent sample."""

        if self.crossover:
            parent1 = self.best_parent()
            parent2 = self.best_parent()
            if set(parent1.keys()) != set(parent2.keys()):
                raise ValueError(f'Parents have different keys: {parent1.keys()} and {parent2.keys()}.')
            # Crossover to get a "parent".
            parent = copy.copy(parent1)
            for key, value in parent2.items():
                # Each dimension has 50% chance to be inherited from parent2.
                if self._random_state.uniform(0, 1) < 0.5:
                    parent[key] = value
        else:
            parent = self.best_parent()

        space = self.model_space.simplify()

        sample = copy.copy(parent)
        for key, mutable in space.items():
            if isinstance(mutable, MutableAnnotation):
                # Skip annotations because resampling them are meaningless.
                continue
            if key not in sample:
                raise KeyError(f'Key {key} not found in parent sample {parent}.')
            if self._random_state.uniform(0, 1) < self.mutation_prob:
                # NOTE: we do not exclude the original choice here for simplicity,
                # which is slightly different from the original paper.
                sample[key] = mutable.random(random_state=self._random_state)

        # Reject duplicate samples here, raise error for retry if duplicate.
        if self._dedup_helper is not None:
            self._dedup_helper.dedup(sample)

        # Reject invalid samples here.
        model = self.model_space.freeze(sample)

        self._individual_counter += 1
        _logger.info('[Individual %4d] Mutated: %s', self._individual_counter, sample)
        return model

    def best_parent(self) -> Sample:
        """Get the best individual from a randomly sampled subset of the population."""
        samples = list(self._population)
        samples = [samples[i] for i in self._random_state.permutation(len(samples))[:self.sample_size]]
        parent = max(samples, key=lambda sample: sample.y).x
        _logger.debug('Parent picked: %s', parent)
        return parent

    def _initialize(self, model_space: ExecutableModelSpace, engine: ExecutionEngine) -> ExecutableModelSpace:
        engine.register_model_event_callback(ModelEventType.TrainingEnd, self._training_end_callback)
        return model_space

    def _cleanup(self) -> None:
        _logger.debug('Unregistering event callbacks...')
        self.engine.unregister_model_event_callback(ModelEventType.TrainingEnd, self._training_end_callback)

    def _run(self) -> None:
        _logger.info('Spawning the initial population. %d individuals to go.', self.population_size - len(self._population))
        while len(self._population) + len(self._running_models) < self.population_size:
            if not self.wait_for_resource():
                return

            model = self._retry_helper.retry(self.random)
            if model is None:
                _logger.warning('Cannot find a new model to submit. Stop.')
                return

            self._running_models.append(model)
            self.engine.submit_models(model)

        # Mutation of models
        _logger.info('Spawning mutated individuals.')
        # Find a resource here.
        # Ideally it should lock the resource (if multiple strategies share one engine).
        while self.wait_for_resource():
            model = self._retry_helper.retry(self.new_individual)
            if model is None:
                _logger.warning('Cannot find a new model to submit. Stop.')
                return

            self._running_models.append(model)
            self.engine.submit_models(model)

        _logger.debug('Waiting for all the models to change status...')
        self.engine.wait_models()  # Wait for the rest of the population.

    def state_dict(self) -> dict:
        dedup_state = self._dedup_helper.state_dict() if self._dedup_helper is not None else {}
        return {
            'population': list(self._population),
            'individual_counter': self._individual_counter,
            'random_state': self._random_state.get_state(),
            'num_running_models': len(self._running_models),
            **dedup_state
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if self._dedup_helper is not None:
            self._dedup_helper.load_state_dict(state_dict)

        if state_dict.get('num_running_models', 0) > 0:
            _logger.warning('Loaded state dict has %d running models. They will be ignored.', state_dict['num_running_models'])
            # TODO: Unfinished models are in the state of dedup, but they shouldn't be deduped.
            #       They should be tried again when the strategy resumes.
        self._population = collections.deque(state_dict['population'])
        self._individual_counter = state_dict['individual_counter']
        self._random_state.set_state(state_dict['random_state'])

    def _training_end_callback(self, event: ModelEvent) -> None:
        # NOTE: It would be better if there's a thread lock here.
        # However, I don't think it will do much harm if we don't have it.
        if event.model in self._running_models:
            self._running_models.remove(event.model)
            if event.model.metric is not None:
                _logger.info('[Metric] %f Sample: %s', event.model.metric, event.model.sample)
                # Even if it fails, as long as it has a metric, we add it to the population.
                assert event.model.sample is not None
                self._population.append(Individual(event.model.sample, event.model.metric))
                _logger.debug('New individual added to population: %s', self._population[-1])
                if len(self._population) > self.population_size:
                    self._population.popleft()
            else:
                _logger.warning('%s has no metric. Skip.', event.model)
        else:
            _logger.warning('%s is not in the running list. Ignore.', event.model)
