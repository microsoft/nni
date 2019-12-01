import logging

import numpy as np
import torch
import torch.nn.functional as F

from nni.nas.pytorch.random import RandomMutator

_logger = logging.getLogger(__name__)


class SPOSSupernetTrainingMutator(RandomMutator):
    def __init__(self, model, flops_func, flops_lb, flops_ub,
                 flops_bin_num=7, flops_sample_timeout=500):
        super().__init__(model)
        self._flops_func = flops_func
        self._flops_bin_num = flops_bin_num
        self._flops_bins = [flops_lb + (flops_ub - flops_lb) / flops_bin_num * i for i in range(flops_bin_num)]
        self._flops_sample_timeout = flops_sample_timeout

    def sample_search(self):
        for _ in range(self._flops_sample_timeout):
            idx = np.random.randint(self._flops_bin_num)
            cand = super().sample_search()
            if self._flops_bins[idx] <= self._flops_func(cand) <= self._flops_bins[idx + 1]:
                _logger.debug("Sampled candidate flops %f.", cand)
                return cand
        _logger.warning("Failed to sample a flops-valid candidate within %d tries.", self._flops_sample_timeout)
        return super().sample_search()

    def sample_final(self):
        return self.sample_search()
