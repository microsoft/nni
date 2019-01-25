# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge,
# to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import sys
import numpy

from scipy.stats import norm
from scipy.optimize import minimize

import nni.metis_tuner.lib_data as lib_data


def next_hyperparameter_expected_improvement(fun_prediction,   
                                             fun_prediction_args,
                                             x_bounds, x_types,
                                             samples_y_aggregation,
                                             minimize_starting_points,
                                             minimize_constraints_fun=None):
    '''
    "Expected Improvement" acquisition function
    '''
    best_x = None
    best_acquisition_value = None
    x_bounds_minmax = [[i[0], i[-1]] for i in x_bounds]
    x_bounds_minmax = numpy.array(x_bounds_minmax)

    for starting_point in numpy.array(minimize_starting_points):
        res = minimize(fun=_expected_improvement,
                       x0=starting_point.reshape(1, -1),
                       bounds=x_bounds_minmax,
                       method="L-BFGS-B",
                       args=(fun_prediction,
                             fun_prediction_args,
                             x_bounds,
                             x_types,
                             samples_y_aggregation,
                             minimize_constraints_fun))

        if (best_acquisition_value is None) or \
                (res.fun < best_acquisition_value):
            res.x = numpy.ndarray.tolist(res.x)
            res.x = lib_data.match_val_type(res.x, x_bounds, x_types)
            if (minimize_constraints_fun is None) or \
                    (minimize_constraints_fun(res.x) is True):
                best_acquisition_value = res.fun
                best_x = res.x

    outputs = None
    if best_x is not None:
        mu, sigma = fun_prediction(best_x, *fun_prediction_args)
        outputs = {'hyperparameter': best_x, 'expected_mu': mu,
                   'expected_sigma': sigma, 'acquisition_func': "ei"}

    return outputs

def _expected_improvement(x, fun_prediction, fun_prediction_args,
                          x_bounds, x_types, samples_y_aggregation,
                          minimize_constraints_fun):
    # This is only for step-wise optimization
    x = lib_data.match_val_type(x, x_bounds, x_types)

    expected_improvement = sys.maxsize
    if (minimize_constraints_fun is None) or (minimize_constraints_fun(x) is True):
        mu, sigma = fun_prediction(x, *fun_prediction_args)

        loss_optimum = min(samples_y_aggregation)
        scaling_factor = -1

        # In case sigma equals zero
        with numpy.errstate(divide="ignore"):
            Z = scaling_factor * (mu - loss_optimum) / sigma
            expected_improvement = scaling_factor * (mu - loss_optimum) * \
                                        norm.cdf(Z) + sigma * norm.pdf(Z)
            expected_improvement = 0.0 if sigma == 0.0 else expected_improvement

        # We want expected_improvement to be as large as possible
        # (i.e., as small as possible for minimize(...))
        expected_improvement = -1 * expected_improvement
    return expected_improvement


def next_hyperparameter_lowest_confidence(fun_prediction,
                                          fun_prediction_args,
                                          x_bounds, x_types,
                                          minimize_starting_points,
                                          minimize_constraints_fun=None):
    '''
    "Lowest Confidence" acquisition function
    '''
    best_x = None
    best_acquisition_value = None
    x_bounds_minmax = [[i[0], i[-1]] for i in x_bounds]
    x_bounds_minmax = numpy.array(x_bounds_minmax)

    for starting_point in numpy.array(minimize_starting_points):
        res = minimize(fun=_lowest_confidence,
                       x0=starting_point.reshape(1, -1),
                       bounds=x_bounds_minmax,
                       method="L-BFGS-B",
                       args=(fun_prediction,
                             fun_prediction_args,
                             x_bounds,
                             x_types,
                             minimize_constraints_fun))

        if (best_acquisition_value) is None or (res.fun < best_acquisition_value):
            res.x = numpy.ndarray.tolist(res.x)
            res.x = lib_data.match_val_type(res.x, x_bounds, x_types)
            if (minimize_constraints_fun is None) or (minimize_constraints_fun(res.x) is True):
                best_acquisition_value = res.fun
                best_x = res.x

    outputs = None
    if best_x is not None:
        mu, sigma = fun_prediction(best_x, *fun_prediction_args)
        outputs = {'hyperparameter': best_x, 'expected_mu': mu,
                   'expected_sigma': sigma, 'acquisition_func': "lc"}
    return outputs

def _lowest_confidence(x, fun_prediction, fun_prediction_args,
                       x_bounds, x_types, minimize_constraints_fun):
    # This is only for step-wise optimization
    x = lib_data.match_val_type(x, x_bounds, x_types)

    ci = sys.maxsize
    if (minimize_constraints_fun is None) or (minimize_constraints_fun(x) is True):
        mu, sigma = fun_prediction(x, *fun_prediction_args)
        ci = (sigma * 1.96 * 2) / mu
        # We want ci to be as large as possible
        # (i.e., as small as possible for minimize(...),
        # because this would mean lowest confidence
        ci = -1 * ci

    return ci


def next_hyperparameter_lowest_mu(fun_prediction,
                                  fun_prediction_args,
                                  x_bounds, x_types,
                                  minimize_starting_points,
                                  minimize_constraints_fun=None):
    '''
    "Lowest Mu" acquisition function
    '''
    best_x = None
    best_acquisition_value = None
    x_bounds_minmax = [[i[0], i[-1]] for i in x_bounds]
    x_bounds_minmax = numpy.array(x_bounds_minmax)

    for starting_point in numpy.array(minimize_starting_points):
        res = minimize(fun=_lowest_mu,
                       x0=starting_point.reshape(1, -1),
                       bounds=x_bounds_minmax,
                       method="L-BFGS-B",
                       args=(fun_prediction, fun_prediction_args, \
                             x_bounds, x_types, minimize_constraints_fun))

        if (best_acquisition_value is None) or (res.fun < best_acquisition_value):
            res.x = numpy.ndarray.tolist(res.x)
            res.x = lib_data.match_val_type(res.x, x_bounds, x_types)
            if (minimize_constraints_fun is None) or (minimize_constraints_fun(res.x) is True):
                best_acquisition_value = res.fun
                best_x = res.x

    outputs = None
    if best_x is not None:
        mu, sigma = fun_prediction(best_x, *fun_prediction_args)
        outputs = {'hyperparameter': best_x, 'expected_mu': mu,
                   'expected_sigma': sigma, 'acquisition_func': "lm"}
    return outputs


def _lowest_mu(x, fun_prediction, fun_prediction_args,
               x_bounds, x_types, minimize_constraints_fun):
    '''
    Calculate the lowest mu
    '''
    # This is only for step-wise optimization
    x = lib_data.match_val_type(x, x_bounds, x_types)

    mu = sys.maxsize
    if (minimize_constraints_fun is None) or (minimize_constraints_fun(x) is True):
        mu, _ = fun_prediction(x, *fun_prediction_args)
    return mu
    