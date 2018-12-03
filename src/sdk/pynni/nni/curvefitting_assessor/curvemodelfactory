# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the 'Software'), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import sys
import os
sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]) 

import logging
import numpy as np
import emcee
import inspect
import traceback
from functools import reduce
from scipy.misc import logsumexp
from scipy.stats import norm, kde
from scipy.optimize import curve_fit, leastsq, fmin_bfgs, fmin_l_bfgs_b, nnls
from curvefunctions import curve_combination_models, model_defaults

logger = logging.getLogger('curvefitting_Assessor: modelfactory')

def recency_weights(num):
    if num == 1:
        return np.ones(1)
    else:
        recency_weights = [10**(1./num)] * num
        recency_weights = recency_weights**(np.arange(0, num))
        return recency_weights
    
class CurveModel(object):
    def __init__(self, function, function_der=None, min_vals={}, max_vals={}, default_vals={}):
        '''
        function: the function to be fit
        function_der: derivative of the function
        '''
        self.function = function
        if function_der != None:
            logger.warning('function derivate is not implemented yet...sorry!')
        self.function_der = function_der
        assert isinstance(min_vals, dict)
        self.min_vals = min_vals.copy()
        assert isinstance(max_vals, dict)
        self.max_vals = max_vals.copy()
        function_args = inspect.getargspec(function).args
        assert 'x' in function_args, 'The function needs \'x\' as a parameter.'
        for default_param_name in default_vals.keys():
            if default_param_name == 'sigma':
                continue
            logger.warning('function %s doesn\'t take default param %s', function.__name__, default_param_name)
        self.function_params = [param for param in function_args if param != 'x']
        self.default_vals = default_vals.copy()
        for param_name in self.function_params:
            if param_name not in default_vals:
                self.default_vals[param_name] = 1.0
        self.all_param_names = [param for param in self.function_params]
        self.all_param_names.append('sigma')
        self.name = self.function.__name__
        self.ndim = len(self.all_param_names)
        self.ml_params = None

        #uniform noise prior over interval:
        if 'sigma' not in self.min_vals:
            self.min_vals['sigma'] = 0.
        if 'sigma' not in self.max_vals:
            self.max_vals['sigma'] = 1.0
        if 'sigma' not in self.default_vals:
            self.default_vals['sigma'] = 0.05

    def default_function_param_array(self):
        return np.asarray([self.default_vals[param_name] for param_name in self.function_params])

    def are_params_in_bounds(self, theta):
        '''
            Are the parameters in their respective bounds?
        '''
        in_bounds = True
    
        for param_name, param_value in zip(self.all_param_names, theta):
            if param_name in self.min_vals:
                if param_value < self.min_vals[param_name]:
                    in_bounds = False
            if param_name in self.max_vals:
                if param_value > self.max_vals[param_name]:
                    in_bounds = False
        return in_bounds

    def split_theta(self, theta):
        '''Split theta into the function parameters (dict) and sigma. '''
        params = {}
        sigma = None
        for param_name, param_value in zip(self.all_param_names, theta):
            if param_name in self.function_params:
                params[param_name] = param_value
            elif param_name == 'sigma':
                sigma = param_value
        return params, sigma

    def split_theta_to_array(self, theta):
        '''Split theta into the function parameters (array) and sigma. '''
        params = theta[:-1]
        sigma = theta[-1]
        return params, sigma

    def fit(self, x, y):
        raise NotImplementedError()

    def predict(self, x):
        raise NotImplementedError()

    def predict_given_theta(self, x, theta):
        '''
            Make predictions given a single theta
        '''
        params, sigma = self.split_theta(theta)
        predictive_mu = self.function(x, **params)
        return predictive_mu, sigma

    def likelihood(self, x, y):
        '''
            for each y_i in y:
                p(y_i|x, model)
        '''
        params, sigma = self.split_theta(self.ml_params)
        return norm.pdf(y-self.function(x, **params), loc=0, scale=sigma)

class MLCurveModel(CurveModel):
    '''
    ML fit of a curve.
    '''
    def __init__(self, recency_weighting=True,  **kwargs):
        super(MLCurveModel, self).__init__(**kwargs)

        #Maximum Likelihood values of the parameters
        self.recency_weighting = recency_weighting

    def fit(self, x, y, weights=None, start_from_default=True):
        '''
        weights: None or weight for each sample.
        non-linear least-squares fit of the data.
        First tries Levenberg-Marquardt and falls back
        to BFGS in case that fails.
        Start from default values or from previous ml_params?
        '''
        if self.fit_leastsq(x, y, weights, start_from_default) or self.fit_bfgs(x, y, weights, start_from_default):
            return True
        else:
            return False

    def predict(self, x):
        #assert len(x.shape) == 1
        params, sigma = self.split_theta_to_array(self.ml_params)
        return self.function(x, *params)
        #return np.asarray([self.function(x_pred, **params) for x_pred in x])

    def fit_ml(self, x, y, weights, start_from_default):
        '''
        non-linear least-squares fit of the data.

        First tries Levenberg-Marquardt and falls back
        to BFGS in case that fails.

        Start from default values or from previous ml_params?
        '''
        successful = self.fit_leastsq(x, y, weights, start_from_default)
        if not successful:
            successful = self.fit_bfgs(x, y, weights, start_from_default)
            if not successful:
                return False
        return successful

    def ml_sigma(self, x, y, popt, weights):
        '''
        Given the ML parameters (popt) get the ML estimate of sigma.
        '''
        if weights is None:
            if self.recency_weighting:
                variance = np.average((y-self.function(x, *popt))**2,
                    weights=recency_weights(len(y)))
                sigma = np.sqrt(variance)
            else:
                sigma = (y-self.function(x, *popt)).std()
        else:
            if self.recency_weighting:
                variance = np.average((y-self.function(x, *popt))**2,
                    weights=recency_weights(len(y)) * weights)
                sigma = np.sqrt(variance)
            else:
                variance = np.average((y-self.function(x, *popt))**2,
                    weights=weights)
                sigma = np.sqrt(variance)
        return sigma

    def fit_leastsq(self, x, y, weights, start_from_default):    
        try:
            if weights is None:
                if self.recency_weighting:
                    residuals = lambda p: np.sqrt(recency_weights(len(y))) * (self.function(x, *p) - y)
                else:
                    residuals = lambda p: self.function(x, *p) - y
            else:
                #the return value of this function will be squared, hence
                #we need to take the sqrt of the weights here
                if self.recency_weighting:
                    residuals = lambda p: np.sqrt(recency_weights(len(y))*weights) * (self.function(x, *p) - y)
                else:
                    residuals = lambda p: np.sqrt(weights) * (self.function(x, *p) - y)

            if start_from_default:
                initial_params = self.default_function_param_array()
            else:
                initial_params, _ = self.split_theta_to_array(self.ml_params)

            popt, cov_popt, info, msg, status = leastsq(residuals,
                    x0=initial_params,
                    full_output=True)
                #Dfun=,
                #col_deriv=True)
            if np.any(np.isnan(info['fjac'])):
                return False

            leastsq_success_statuses = [1,2,3,4]
            if status in leastsq_success_statuses:
                if any(np.isnan(popt)):
                    return False
                #within bounds?
                if not self.are_params_in_bounds(popt):
                    return False

                sigma = self.ml_sigma(x, y, popt, weights)
                self.ml_params = np.append(popt, [sigma])

                logging.info('leastsq successful for model %s' % self.function.__name__)

                return True
            else:
                logging.warn('leastsq NOT successful for model %s, msg: %s' % (self.function.__name__, msg))
                logging.warn('best parameters found: ' + str(popt))
                return False
        except Exception as e:
            logger.warning(traceback.format_exc())
            return False

    def fit_bfgs(self, x, y, weights, start_from_default):
        try:
            def objective(params):
                if weights is None:
                    if self.recency_weighting:
                        return np.sum(recency_weights(len(y))*(self.function(x, *params) - y)**2)
                    else:
                        return np.sum((self.function(x, *params) - y)**2)
                else:
                    if self.recency_weighting:
                        return np.sum(weights * recency_weights(len(y)) * (self.function(x, *params) - y)**2)
                    else:
                        return np.sum(weights * (self.function(x, *params) - y)**2)
            bounds = []
            for param_name in self.function_params:
                if param_name in self.min_vals and param_name in self.max_vals:
                    bounds.append((self.min_vals[param_name], self.max_vals[param_name]))
                elif param_name in self.min_vals:
                    bounds.append((self.min_vals[param_name], None))
                elif param_name in self.max_vals:
                    bounds.append((None, self.max_vals[param_name]))
                else:
                    bounds.append((None, None))

            if start_from_default:
                initial_params = self.default_function_param_array()
            else:
                initial_params, _ = self.split_theta_to_array(self.ml_params)

            popt, fval, info= fmin_l_bfgs_b(objective,
                                            x0=initial_params,
                                            bounds=bounds,
                                            approx_grad=True)
            if info['warnflag'] != 0:
                logging.warn('BFGS not converged! (warnflag %d) for model %s' % (info['warnflag'], self.name))
                logging.warn(info)
                return False

            if popt is None:
                return False
            if any(np.isnan(popt)):
                logging.info('bfgs NOT successful for model %s, parameter NaN' % self.name)
                return False
            sigma = self.ml_sigma(x, y, popt, weights)
            self.ml_params = np.append(popt, [sigma])
            logging.info('bfgs successful for model %s' % self.name)
            return True
        except:
            return False

    def aic(self, x, y):
        '''
        Akaike information criterion
        http://en.wikipedia.org/wiki/Akaike_information_criterion
        '''
        params, sigma = self.split_theta_to_array(self.ml_params)
        y_model = self.function(x, *params)
        log_likelihood = norm.logpdf(y-y_model, loc=0, scale=sigma).sum()
        return 2 * len(self.function_params) - 2 * log_likelihood
