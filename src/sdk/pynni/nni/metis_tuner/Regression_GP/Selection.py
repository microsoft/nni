# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import random
import sys

import nni.metis_tuner.lib_acquisition_function as lib_acquisition_function
import nni.metis_tuner.lib_constraint_summation as lib_constraint_summation
import nni.metis_tuner.lib_data as lib_data
import nni.metis_tuner.Regression_GP.Prediction as gp_prediction

sys.path.insert(1, os.path.join(sys.path[0], '..'))

CONSTRAINT_LOWERBOUND = None
CONSTRAINT_UPPERBOUND = None
CONSTRAINT_PARAMS_IDX = []


def selection_r(acquisition_function,
                samples_y_aggregation,
                x_bounds,
                x_types,
                regressor_gp,
                num_starting_points=100,
                minimize_constraints_fun=None):
    '''
    Selecte R value
    '''
    minimize_starting_points = [lib_data.rand(x_bounds, x_types) \
                                    for i in range(0, num_starting_points)]
    outputs = selection(acquisition_function, samples_y_aggregation,
                        x_bounds, x_types, regressor_gp,
                        minimize_starting_points,
                        minimize_constraints_fun=minimize_constraints_fun)

    return outputs

def selection(acquisition_function,
              samples_y_aggregation,
              x_bounds, x_types,
              regressor_gp,
              minimize_starting_points,
              minimize_constraints_fun=None):
    '''
    selection
    '''
    outputs = None

    sys.stderr.write("[%s] Exercise \"%s\" acquisition function\n" \
                        % (os.path.basename(__file__), acquisition_function))

    if acquisition_function == "ei":
        outputs = lib_acquisition_function.next_hyperparameter_expected_improvement(\
                        gp_prediction.predict, [regressor_gp], x_bounds, x_types, \
                        samples_y_aggregation, minimize_starting_points, \
                        minimize_constraints_fun=minimize_constraints_fun)
    elif acquisition_function == "lc":
        outputs = lib_acquisition_function.next_hyperparameter_lowest_confidence(\
                        gp_prediction.predict, [regressor_gp], x_bounds, x_types,\
                        minimize_starting_points, minimize_constraints_fun=minimize_constraints_fun)
    elif acquisition_function == "lm":
        outputs = lib_acquisition_function.next_hyperparameter_lowest_mu(\
                        gp_prediction.predict, [regressor_gp], x_bounds, x_types,\
                        minimize_starting_points, minimize_constraints_fun=minimize_constraints_fun)
    return outputs

def _rand_with_constraints(x_bounds, x_types):
    '''
    Random generate with constraints
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
    Minimize the constraints fun summation
    '''
    summation = sum([x[i] for i in CONSTRAINT_PARAMS_IDX])
    return CONSTRAINT_UPPERBOUND >= summation >= CONSTRAINT_LOWERBOUND
