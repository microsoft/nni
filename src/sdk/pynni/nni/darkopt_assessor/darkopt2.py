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

import numpy as np
import pymc3
import scipy.stats

class EnsembleSamplingPredictor(object):
    def __init__(self):
        self.traces = None

    def fit(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        self.traces = sample_ensemble(x, y)

    def predict_proba_less_than(self, x, y):
        return np.mean([
            predict_proba_less_than_ensemble(x, y, trace)
            for trace in self.traces
        ], axis=0)


curves = [
    ('vapore_pressure', 3, lambda x, p: p[0] * np.exp(p[1] / (1 + x) + p[2] * np.log1p(x))),
    ('weibull', 3, lambda x, p: p[0] - p[1] * np.exp(-p[2] * x)),
]


def _single(x, y, curve):
    name, n_params, func = curve
    with pymc3.Model() as model_single:
        params = pymc3.Flat(name, shape=n_params)
        mu = func(x, params)
        sd = pymc3.Uniform('sd', lower=1e-9, upper=1e-1)
        pymc3.Normal('y_obs', mu=mu, sd=sd, observed=y)
        map_estimate = pymc3.find_MAP()
        return map_estimate[name]


def sample_ensemble(x, y):
    start = { curve[0]: _single(x, y, curve) for curve in curves }
    start['weights_unnormalized_interval_'] = np.zeros(len(curves))
    start['sd_interval_'] = 0
    with pymc3.Model() as model_ensemble:
        mu_single = []
        for name, n_params, func in curves:
            params = pymc3.Flat(name, shape=n_params)
            mu_single.append(func(x, params))
        weights_unnormalized = pymc3.Uniform(
            'weights_unnnormalized', lower=0, upper=1, shape=len(curves))
        weights_normalized = pymc3.Deterministic(
            'weights_normalized', weights_unnormalized / weights_unnormalized.sum())
        mu_ensemble = weights_normalized.dot(mu_single)
        sd1 = pymc3.Uniform('sd', lower=1e-9, upper=1e-1)
        pymc3.Deterministic('sd1', sd1)
        pymc3.Normal('y_obs', mu=mu_ensemble, observed=y, sd=sd1)
        return pymc3.sample(start=start, step=pymc3.Metropolis(), draws=1000)


def predict_proba_less_than_ensemble(x, y, param):
    ps = [func(x, param[name]) for name, _, func in curves]
    mu = param['weights_normalized'].dot(ps)
    sd1 = param['sd1']
    return scipy.stats.norm.cdf(y, loc=mu, scale=sd1)