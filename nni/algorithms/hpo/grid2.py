# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Grid search tuner for hyper-parameter optimization.

This tuner does not support normal distributions.

For categorical parameters this tuner fully explore all combinations.
For numerical parameters it samples them at progressively decreased intervals.

Use this tuner if you have abundant resource and want to find strictly optimal parameters.

Grid search tuner has no argument.
"""

__all__ = 'GridSearchTuner'

import logging

import numpy as np

import nni
from nni.common.hpo_utils import ParameterSpec, deformat_parameters, deformat_single_parameter, format_search_space

_logger = logging.getLogger('nni.tuner.gridsearch')

##
# Grid search is a simple algorithm if only categorical parameters are considered.
# But to support continuous space, things get tricky.
#
# To support continuous space, we divide search process into "epochs".
# The first epoch only explores lowest and highest point of uniform parameters.
# When first epoch is fully explored, the algorithm starts second epoch,
# where it divides uniform spaces by adding middle points into the grid.
# Then in third epoch it adds quartile points, and so on.
# Of course, the algorithm will skip parameters already tried in previous epochs.
#
# There is another problem, "q".
# We do not want to convert quniform/qloguniform to choices,
# because large spaces like "qloguniform(1, 1000000, 1)" will become un-searchable.
# And we do not want to ignore the "q", or otherwise small spaces can cause exponential explosion.
# To solve this, the algorithm will eliminate undivisible "q" ranges at the end of each epoch.
#
# Here is an example:
#
#   search space:
#     x: choices(5, 7)
#     y: uniform(-1, 1)
#     z: quniform(2, 5, 1)
#
#   grid of first epoch:
#     x: [5, 7]
#     y: [-1, 1]
#     z: [2, 5]
#   generated parameters:
#     (5,-1,2) (5,-1,5)  (5,1,2) (5,1,5)   (7,-1,2) (7,-1,5)  (7,1,2) (7,1,5)
#
#   grid of second epoch:
#     x: [5, 7]
#     y: [-1, 1, 0]
#     z: [2, 5, 3.5]  (results in [2, 4, 5])
#   generated parameters:
#     (5,-1,4)  (5,1,4)  (5,0,2) (5,0,5) (5,0,4)
#     (7,-1,4)  (7,1,4)  (7,0,2) (7,0,5) (7,0,4)
#
#   grid of third epoch:
#     x: [5, 7]
#     y: [-1, 0, 1, -0.5, 0.5]  (old values are sorted in the implementation)
#     z: [2, 3.5, 5, 2.75]  (results in [2, 3, 4, 5])
#   generated parameters:
#     (5,-1,3)  (5,0,3)  (5,1,3)  (5,-0.5,2) (5,-0.5,4) (5,-0.5,5) (5,-0.5,3)  (5,0.5,2) (5,0.5,4) (5,0.5,5) (5,0.5,3)
#     (7,-1,3)  (7,0,3)  (7,1,3)  (7,-0.5,2) (7,-0.5,4) (7,-0.5,5) (7,-0.5,3)  (7,0.5,2) (7,0.5,4) (7,0.5,5) (7,0.5,3)
##

class GridSearchTuner(nni.tuner.Tuner):
    def __init__(self):
        self.space = None

        # the grid to search in this epoch
        # when the space is fully explored, grid is set to None
        self.grid = None  # list[int | float]

        # a paremter set is internally expressed as a vector
        # for each dimension i, self.vector[i] is the parameter value's index in self.grid[i]
        # in third epoch of above example, vector [1, 3, 0] means parameters {x: 7, y: -0.5, z: 2}
        self.vector = None  # list[int]

        # this tells which parameter candidates are newly added in current epoch
        # during third epoch of above example, epoch_bar is [2, 3, 3]
        self.epoch_bar = None  # list[int]

        # for "q" parameters, this stores which ranges may be divisible
        # at the end of each epoch in above example, the value will be:
        #   1st: {2: [(2, 5)]}
        #   2nd: {2: [(2, 3.5), (3.5, 5)]}
        #   3rd: {2: [(2, 2.75), (2.75, 3.5), (4.25, 5)]}  (3.5~4.25 is eliminated)
        self.divisions = None  # dict[int, list[tuple[float, float]]]

    def update_search_space(self, space):
        self.space = format_search_space(space).values()
        if not self.space:  # the tuner will crash in this case, report it explicitly
            raise ValueError('Grid search tuner does not support empty search space')
        if any(spec.normal_distributed for spec in self.space):
            raise NotImplementedError('Grid search does not support normal distribution')
        self._init_grid()

    def generate_parameters(self, *args, **kwargs):
        params = self._suggest()
        if params is None:  # fully explored
            raise nni.NoMoreTrialError
        return deformat_parameters(params, self.space)

    def receive_trial_result(self, *args, **kwargs):
        pass

    def _suggest(self):
        # returns next parameter set, or None if the space is already fully explored
        while True:
            if self.grid is None:
                # search space fully explored
                return None

            self._next_vector()

            if self.vector is None:
                # epoch end, update grid and retry
                self._next_grid()
                continue

            if all((self.vector[i] < self.epoch_bar[i]) for i in range(len(self.space))):
                # already explored in previous epochs
                continue

            # this vector is valid, stop
            return self._current_parameters()

    def _next_vector(self):
        # iterate to next vector of this epoch, set vector to None if epoch end
        if self.vector is None:  # first vector in this epoch
            self.vector = [0] * len(self.space)
            return

        # deals with nested choice, don't touch nested spaces that are not chosen by current vector
        activated_dims = []
        params = self._current_parameters()
        for i, spec in enumerate(self.space.values()):
            if spec.is_activated_in(params):
                activated_dims.append(i)

        for i in reversed(activated_dims):
            if self.vector[i] + 1 < len(self.grid[i]):
                self.vector[i] += 1
                return
            else:
                self.vector[i] = 0

        self.vector = None  # the loop ends without returning, no more vector in this epoch

    def _next_grid(self):
        # update grid information (grid, epoch_bar, divisions) for next epoch
        updated = False
        for i, spec in enumerate(self.space.values()):
            self.epoch_bar[i] = len(self.grid[i])
            if spec.categorical:
                continue
            self.grid[i] = sorted(self.grid[i])

            if spec.q is None:
                for j in range(self.epoch_bar[i] - 1):
                    mid = (self.grid[i][j] + self.grid[i][j + 1]) / 2
                    self.grid[i].append(mid)
            else:
                new_vals = []
                new_divs = []
                for l, r in self.divisions[i]:
                    mid = (l + r) / 2
                    diff_l = _less_after_q(l, mid, spec)
                    diff_r = _less_after_q(mid, r, spec)
                    if diff_l and diff_r:
                        new_vals.append(mid)
                        updated = True
                    if diff_l:
                        new_divs.append((l, mid))
                    if diff_r:
                        new_divs.append((mid, r))
                self.grid[i] += new_vals
                self.divisions[i] = new_divs

        if not updated:  # fully explored
            _logger.info('Search space has been fully explored')
            self.grid = None
        else:
            size = _grid_size_info(self.grid)
            _logger.info('Grid has been extended, new size: {size}')

    def _init_grid(self):
        for i, spec in enumerate(self.space.values()):
            self.epoch_bar[i] = 0
            if spec.categorical:
                self.grid[i] = list(range(spec.size))
                continue

            if spec.q is None:
                self.grid[i] = [spec.low, spec.high]
            else:
                if _less_after_q(spec.low, spec.high, spec):
                    self.grid[i] = [spec.low, spec.high]
                    self.divisions[i] = [(spec.low, spec.high)]
                else:  # only one choice
                    self.grid[i] = [spec.low]
                    self.divisions[i] = []
        _logger.info('G

    def _current_parameters(self):
        # convert self.vector to "formatted" parameters
        params = {}
        for i, spec in enumerate(self.space.values()):
            params[spec.key] = self.grid[i][self.vector[i]]
        return params

def _less_after_q(x, y, spec):
    real_x = _deformat_single_parameter(x, spec)
    real_y = _deformat_single_parameter(y, spec)
    return x < y

def _deformat_single_parameter(x, spec):
    spec_dict = spec._asdict()
    spec_dict['key'] = (spec.name,)
    spec = ParameterSpec(**spec_dict)
    params = deformat_parameter(x, {spec.key: spec})
    return params[spec.key]

def _grid_size_info(grid):
    sizes = [len(candidates) for candidates in grid]
    mul = 'x'.join(str(s) for s in sizes)
    total = np.prod(sizes)
    _logger.info(f'{mul} = {total}')
