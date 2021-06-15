# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import itertools
import logging
import random
import time
from typing import Any, Dict, List

from .. import Sampler, submit_models, query_available_resources, budget_exhausted
from .base import BaseStrategy
from .utils import dry_run_for_search_space, get_targeted_model

_logger = logging.getLogger(__name__)


def grid_generator(search_space: Dict[Any, List[Any]], shuffle=True):
    keys = list(search_space.keys())
    search_space_values = copy.deepcopy(list(search_space.values()))
    if shuffle:
        for values in search_space_values:
            random.shuffle(values)
    for values in itertools.product(*search_space_values):
        yield {key: value for key, value in zip(keys, values)}


def random_generator(search_space: Dict[Any, List[Any]], dedup=True, retries=500):
    keys = list(search_space.keys())
    history = set()
    search_space_values = copy.deepcopy(list(search_space.values()))
    while True:
        for retry_count in range(retries):
            selected = [random.choice(v) for v in search_space_values]
            if not dedup:
                break
            selected = tuple(selected)
            if selected not in history:
                history.add(selected)
                break
            if retry_count + 1 == retries:
                _logger.debug('Random generation has run out of patience. There is nothing to search. Exiting.')
                return
        yield {key: value for key, value in zip(keys, selected)}


class GridSearch(BaseStrategy):
    """
    Traverse the search space and try all the possible combinations one by one.

    Parameters
    ----------
    shuffle : bool
        Shuffle the order in a candidate list, so that they are tried in a random order. Default: true.
    """

    def __init__(self, shuffle=True):
        self._polling_interval = 2.
        self.shuffle = shuffle

    def run(self, base_model, applied_mutators):
        search_space = dry_run_for_search_space(base_model, applied_mutators)
        for sample in grid_generator(search_space, shuffle=self.shuffle):
            _logger.debug('New model created. Waiting for resource. %s', str(sample))
            while query_available_resources() <= 0:
                if budget_exhausted():
                    return
                time.sleep(self._polling_interval)
            submit_models(get_targeted_model(base_model, applied_mutators, sample))


class _RandomSampler(Sampler):
    def choice(self, candidates, mutator, model, index):
        return random.choice(candidates)


class Random(BaseStrategy):
    """
    Random search on the search space.

    Parameters
    ----------
    variational : bool
        Do not dry run to get the full search space. Used when the search space has variational size or candidates. Default: false.
    dedup : bool
        Do not try the same configuration twice. When variational is true, deduplication is not supported. Default: true.
    """

    def __init__(self, variational=False, dedup=True):
        self.variational = variational
        self.dedup = dedup
        if variational and dedup:
            raise ValueError('Dedup is not supported in variational mode.')
        self.random_sampler = _RandomSampler()
        self._polling_interval = 2.

    def run(self, base_model, applied_mutators):
        if self.variational:
            _logger.info('Random search running in variational mode.')
            sampler = _RandomSampler()
            for mutator in applied_mutators:
                mutator.bind_sampler(sampler)
            while True:
                avail_resource = query_available_resources()
                if avail_resource > 0:
                    model = base_model
                    for mutator in applied_mutators:
                        model = mutator.apply(model)
                    _logger.debug('New model created. Applied mutators are: %s', str(applied_mutators))
                    submit_models(model)
                elif budget_exhausted():
                    break
                else:
                    time.sleep(self._polling_interval)
        else:
            _logger.info('Random search running in fixed size mode. Dedup: %s.', 'on' if self.dedup else 'off')
            search_space = dry_run_for_search_space(base_model, applied_mutators)
            for sample in random_generator(search_space, dedup=self.dedup):
                _logger.debug('New model created. Waiting for resource. %s', str(sample))
                while query_available_resources() <= 0:
                    if budget_exhausted():
                        return
                    time.sleep(self._polling_interval)
                submit_models(get_targeted_model(base_model, applied_mutators, sample))
