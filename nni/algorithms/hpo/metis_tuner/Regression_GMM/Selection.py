# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import random
import sys

from .. import lib_acquisition_function
from .. import lib_constraint_summation

sys.path.insert(1, os.path.join(sys.path[0], '..'))


CONSTRAINT_LOWERBOUND = None
CONSTRAINT_UPPERBOUND = None
CONSTRAINT_PARAMS_IDX = []


def _ratio_scores(parameters_value, clusteringmodel_gmm_good,
                  clusteringmodel_gmm_bad):
    '''
    The ratio is smaller the better
    '''
    ratio = clusteringmodel_gmm_good.score(
        [parameters_value]) / clusteringmodel_gmm_bad.score([parameters_value])
    sigma = 0
    return ratio, sigma


def selection_r(x_bounds,
                x_types,
                clusteringmodel_gmm_good,
                clusteringmodel_gmm_bad,
                num_starting_points=100,
                minimize_constraints_fun=None):
    '''
    Select using different types.
    '''
    minimize_starting_points = clusteringmodel_gmm_good.sample(n_samples=num_starting_points)

    outputs = selection(x_bounds, x_types,
                        clusteringmodel_gmm_good,
                        clusteringmodel_gmm_bad,
                        minimize_starting_points[0],
                        minimize_constraints_fun)

    return outputs


def selection(x_bounds,
              x_types,
              clusteringmodel_gmm_good,
              clusteringmodel_gmm_bad,
              minimize_starting_points,
              minimize_constraints_fun=None):
    '''
    Select the lowest mu value
    '''
    results = lib_acquisition_function.next_hyperparameter_lowest_mu(
        _ratio_scores, [clusteringmodel_gmm_good, clusteringmodel_gmm_bad],
        x_bounds, x_types, minimize_starting_points,
        minimize_constraints_fun=minimize_constraints_fun)

    return results


def _rand_with_constraints(x_bounds, x_types):
    '''
    Random generate the variable with constraints
    '''
    outputs = None
    x_bounds_withconstraints = [x_bounds[i] for i in CONSTRAINT_PARAMS_IDX]
    x_types_withconstraints = [x_types[i] for i in CONSTRAINT_PARAMS_IDX]
    x_val_withconstraints = lib_constraint_summation.rand(x_bounds_withconstraints,
                                                          x_types_withconstraints,
                                                          CONSTRAINT_LOWERBOUND,
                                                          CONSTRAINT_UPPERBOUND)
    if x_val_withconstraints is not None:
        outputs = [None] * len(x_bounds)
        for i, _ in enumerate(CONSTRAINT_PARAMS_IDX):
            outputs[CONSTRAINT_PARAMS_IDX[i]] = x_val_withconstraints[i]
        for i, _ in enumerate(outputs):
            if outputs[i] is None:
                outputs[i] = random.randint(x_bounds[i][0], x_bounds[i][1])
    return outputs


def _minimize_constraints_fun_summation(x):
    '''
    Minimize constraints fun summation
    '''
    summation = sum([x[i] for i in CONSTRAINT_PARAMS_IDX])
    return CONSTRAINT_UPPERBOUND >= summation >= CONSTRAINT_LOWERBOUND
