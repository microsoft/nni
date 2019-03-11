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


import argparse, json, os, sys
from multiprocessing.dummy import Pool as ThreadPool

import nni.metis_tuner.Regression_GP.CreateModel as gp_create_model
import nni.metis_tuner.Regression_GP.Prediction as gp_prediction
import nni.metis_tuner.lib_data as lib_data

sys.path.insert(1, os.path.join(sys.path[0], '..'))


def _outlierDetection_threaded(inputs):
    '''
    Detect the outlier
    '''
    [samples_idx, samples_x, samples_y_aggregation] = inputs
    sys.stderr.write("[%s] DEBUG: Evaluating %dth of %d samples\n"\
                        % (os.path.basename(__file__), samples_idx + 1, len(samples_x)))
    outlier = None

    # Create a diagnostic regression model which removes the sample that we want to evaluate
    diagnostic_regressor_gp = gp_create_model.create_model(\
                                    samples_x[0:samples_idx] + samples_x[samples_idx + 1:],\
                                    samples_y_aggregation[0:samples_idx] + samples_y_aggregation[samples_idx + 1:])
    mu, sigma = gp_prediction.predict(samples_x[samples_idx], diagnostic_regressor_gp['model'])

    # 2.33 is the z-score for 98% confidence level
    if abs(samples_y_aggregation[samples_idx] - mu) > (2.33 * sigma):
        outlier = {"samples_idx": samples_idx,
                   "expected_mu": mu,
                   "expected_sigma": sigma,
                   "difference": abs(samples_y_aggregation[samples_idx] - mu) - (2.33 * sigma)}
    return outlier

def outlierDetection_threaded(samples_x, samples_y_aggregation):
    '''
    Use Multi-thread to detect the outlier
    '''
    outliers = []

    threads_inputs = [[samples_idx, samples_x, samples_y_aggregation]\
                            for samples_idx in range(0, len(samples_x))]
    threads_pool = ThreadPool(min(4, len(threads_inputs)))
    threads_results = threads_pool.map(_outlierDetection_threaded, threads_inputs)
    threads_pool.close()
    threads_pool.join()

    for threads_result in threads_results:
        if threads_result is not None:
            outliers.append(threads_result)
        else:
            print("error here.")

    outliers = None if len(outliers) == 0 else outliers
    return outliers

def outlierDetection(samples_x, samples_y_aggregation):
    '''
    '''
    outliers = []
    for samples_idx in range(0, len(samples_x)):
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
                             "difference": abs(samples_y_aggregation[samples_idx] - mu) - (2.33 * sigma)})

    outliers = None if len(outliers) == 0 else outliers
    return outliers
