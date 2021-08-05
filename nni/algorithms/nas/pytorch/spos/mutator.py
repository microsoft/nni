# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

import numpy as np
from nni.algorithms.nas.pytorch.random import RandomMutator

_logger = logging.getLogger(__name__)


class SPOSSupernetTrainingMutator(RandomMutator):
    """
    A random mutator with flops limit.

    Parameters
    ----------
    model : nn.Module
        PyTorch model.
    flops_func : callable
        Callable that takes a candidate from `sample_search` and returns its candidate. When `flops_func`
        is None, functions related to flops will be deactivated.
    flops_lb : number
        Lower bound of flops.
    flops_ub : number
        Upper bound of flops.
    flops_bin_num : number
        Number of bins divided for the interval of flops to ensure the uniformity. Bigger number will be more
        uniform, but the sampling will be slower.
    flops_sample_timeout : int
        Maximum number of attempts to sample before giving up and use a random candidate.
    """
    def __init__(self, model, flops_func=None, flops_lb=None, flops_ub=None,
                 flops_bin_num=7, flops_sample_timeout=500):

        super().__init__(model)
        self._flops_func = flops_func
        if self._flops_func is not None:
            self._flops_bin_num = flops_bin_num
            self._flops_bins = [flops_lb + (flops_ub - flops_lb) / flops_bin_num * i for i in range(flops_bin_num + 1)]
            self._flops_sample_timeout = flops_sample_timeout

    def sample_search(self):
        """
        Sample a candidate for training. When `flops_func` is not None, candidates will be sampled uniformly
        relative to flops.

        Returns
        -------
        dict
        """
        if self._flops_func is not None:
            for times in range(self._flops_sample_timeout):
                idx = np.random.randint(self._flops_bin_num)
                cand = super().sample_search()
                if self._flops_bins[idx] <= self._flops_func(cand) <= self._flops_bins[idx + 1]:
                    _logger.debug("Sampled candidate flops %f in %d times.", cand, times)
                    return cand
            _logger.warning("Failed to sample a flops-valid candidate within %d tries.", self._flops_sample_timeout)
        return super().sample_search()

    def sample_final(self):
        """
        Implement only to suffice the interface of Mutator.
        """
        return self.sample_search()
