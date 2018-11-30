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
'''
parameter_expression.py
'''

import numpy as np


def choice(options, random_state):
    '''
    options: 1-D array-like or int
    random_state: an object of numpy.random.RandomState
    '''
    return random_state.choice(options)


def randint(upper, random_state):
    '''
    upper: an int that represent an upper bound
    random_state: an object of numpy.random.RandomState
    '''
    return random_state.randint(upper)


def uniform(low, high, random_state):
    '''
    low: an float that represent an lower bound
    high: an float that represent an upper bound
    random_state: an object of numpy.random.RandomState
    '''
    assert high > low, 'Upper bound must be larger than lower bound'
    return random_state.uniform(low, high)


def quniform(low, high, q, random_state):
    '''
    low: an float that represent an lower bound
    high: an float that represent an upper bound
    q: sample step
    random_state: an object of numpy.random.RandomState
    '''
    return np.round(uniform(low, high, random_state) / q) * q


def loguniform(low, high, random_state):
    '''
    low: an float that represent an lower bound
    high: an float that represent an upper bound
    random_state: an object of numpy.random.RandomState
    '''
    assert low > 0, 'Lower bound must be positive'
    return np.exp(uniform(np.log(low), np.log(high), random_state))


def qloguniform(low, high, q, random_state):
    '''
    low: an float that represent an lower bound
    high: an float that represent an upper bound
    q: sample step
    random_state: an object of numpy.random.RandomState
    '''
    return np.round(loguniform(low, high, random_state) / q) * q


def normal(mu, sigma, random_state):
    '''
    The probability density function of the normal distribution,
    first derived by De Moivre and 200 years later by both Gauss and Laplace independently.
    mu: float or array_like of floats
        Mean (“centre”) of the distribution.
    sigma: float or array_like of floats
           Standard deviation (spread or “width”) of the distribution.
    random_state: an object of numpy.random.RandomState
    '''
    return random_state.normal(mu, sigma)


def qnormal(mu, sigma, q, random_state):
    '''
    mu: float or array_like of floats
    sigma: float or array_like of floats
    q: sample step
    random_state: an object of numpy.random.RandomState
    '''
    return np.round(normal(mu, sigma, random_state) / q) * q


def lognormal(mu, sigma, random_state):
    '''
    mu: float or array_like of floats
    sigma: float or array_like of floats
    random_state: an object of numpy.random.RandomState
    '''
    return np.exp(normal(mu, sigma, random_state))


def qlognormal(mu, sigma, q, random_state):
    '''
    mu: float or array_like of floats
    sigma: float or array_like of floats
    q: sample step
    random_state: an object of numpy.random.RandomState
    '''
    return np.round(lognormal(mu, sigma, random_state) / q) * q
