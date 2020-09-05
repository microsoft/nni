# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
OutlierDectection.py
"""

import os
import sys
from multiprocessing.dummy import Pool as ThreadPool

import nni.metis_tuner.Regression_GP.CreateModel as gp_create_model
import nni.metis_tuner.Regression_GP.Prediction as gp_prediction

sys.path.insert(1, os.path.join(sys.path[0], '..'))


def _outlierDetection_threaded(inputs):
    """
    Detect the outlier
    """
    [samples_idx, samples_x, samples_y_aggregation] = inputs
    sys.stderr.write("[%s] DEBUG: Evaluating %dth of %d samples\n"
                     % (os.path.basename(__file__), samples_idx + 1, len(samples_x)))
    outlier = None

    # Create a diagnostic regression model which removes the sample that we
    # want to evaluate
    diagnostic_regressor_gp = gp_create_model.create_model(
        samples_x[0:samples_idx] + samples_x[samples_idx + 1:],
        samples_y_aggregation[0:samples_idx] + samples_y_aggregation[samples_idx + 1:])
    mu, sigma = gp_prediction.predict(
        samples_x[samples_idx], diagnostic_regressor_gp['model'])

    # 2.33 is the z-score for 98% confidence level
    if abs(samples_y_aggregation[samples_idx] - mu) > (2.33 * sigma):
        outlier = {"samples_idx": samples_idx,
                   "expected_mu": mu,
                   "expected_sigma": sigma,
                   "difference": abs(samples_y_aggregation[samples_idx] - mu) - (2.33 * sigma)}
    return outlier


def outlierDetection_threaded(samples_x, samples_y_aggregation):
    """
    Use Multi-thread to detect the outlier
    """
    outliers = []

    threads_inputs = [[samples_idx, samples_x, samples_y_aggregation]
                      for samples_idx in range(0, len(samples_x))]
    threads_pool = ThreadPool(min(4, len(threads_inputs)))
    threads_results = threads_pool.map(
        _outlierDetection_threaded, threads_inputs)
    threads_pool.close()
    threads_pool.join()

    for threads_result in threads_results:
        if threads_result is not None:
            outliers.append(threads_result)
        else:
            print("Error: threads_result is None.")

    outliers = outliers if outliers else None
    return outliers


def outlierDetection(samples_x, samples_y_aggregation):
    outliers = []
    for samples_idx, _ in enumerate(samples_x):
        #sys.stderr.write("[%s] DEBUG: Evaluating %d of %d samples\n"
        #  \ % (os.path.basename(__file__), samples_idx + 1, len(samples_x)))
        diagnostic_regressor_gp = gp_create_model.create_model(\
                                        samples_x[0:samples_idx] + samples_x[samples_idx + 1:],\
                                        samples_y_aggregation[0:samples_idx] + samples_y_aggregation[samples_idx + 1:])
        mu, sigma = gp_prediction.predict(samples_x[samples_idx],
                                          diagnostic_regressor_gp['model'])
        # 2.33 is the z-score for 98% confidence level
        if abs(samples_y_aggregation[samples_idx] - mu) > (2.33 * sigma):
            outliers.append({"samples_idx": samples_idx,
                             "expected_mu": mu,
                             "expected_sigma": sigma,
                             "difference": \
                                abs(samples_y_aggregation[samples_idx] - mu) - (2.33 * sigma)})

    outliers = outliers if outliers else None
    return outliers
