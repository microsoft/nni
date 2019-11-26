# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
import numpy

import sklearn.gaussian_process as gp

sys.path.insert(1, os.path.join(sys.path[0], '..'))


def create_model(samples_x, samples_y_aggregation,
                 n_restarts_optimizer=250, is_white_kernel=False):
    '''
    Trains GP regression model
    '''
    kernel = gp.kernels.ConstantKernel(constant_value=1,
                                       constant_value_bounds=(1e-12, 1e12)) * \
                                                gp.kernels.Matern(nu=1.5)
    if is_white_kernel is True:
        kernel += gp.kernels.WhiteKernel(noise_level=1, noise_level_bounds=(1e-12, 1e12))
    regressor = gp.GaussianProcessRegressor(kernel=kernel,
                                            n_restarts_optimizer=n_restarts_optimizer,
                                            normalize_y=True,
                                            alpha=1e-10)
    regressor.fit(numpy.array(samples_x), numpy.array(samples_y_aggregation))

    model = {}
    model['model'] = regressor
    model['kernel_prior'] = str(kernel)
    model['kernel_posterior'] = str(regressor.kernel_)
    model['model_loglikelihood'] = regressor.log_marginal_likelihood(regressor.kernel_.theta)

    return model
