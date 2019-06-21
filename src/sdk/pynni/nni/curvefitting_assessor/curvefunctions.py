# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np

all_models = {}
model_para = {}
model_para_num = {}

curve_combination_models = ['vap', 'pow3', 'linear', 'logx_linear', 'dr_hill_zero_background', 'log_power', 'pow4', 'mmf',
                            'exp4', 'ilog2', 'weibull', 'janoschek']

def vap(x, a, b, c):
    """Vapor pressure model

    Parameters
    ----------
    x: int
    a: float
    b: float
    c: float

    Returns
    -------
    float
        np.exp(a+b/x+c*np.log(x))
    """
    return np.exp(a+b/x+c*np.log(x))

all_models['vap'] = vap
model_para['vap'] = [-0.622028, -0.470050, 0.042322]
model_para_num['vap'] = 3

def pow3(x, c, a, alpha):
    """pow3

    Parameters
    ----------
    x: int
    c: float
    a: float
    alpha: float

    Returns
    -------
    float
        c - a * x**(-alpha)
    """
    return c - a * x**(-alpha)

all_models['pow3'] = pow3
model_para['pow3'] = [0.84, 0.52, 0.01]
model_para_num['pow3'] = 3

def linear(x, a, b):
    """linear

    Parameters
    ----------
    x: int
    a: float
    b: float

    Returns
    -------
    float
        a*x + b
    """
    return a*x + b

all_models['linear'] = linear
model_para['linear'] = [1., 0]
model_para_num['linear'] = 2

def logx_linear(x, a, b):
    """logx linear

    Parameters
    ----------
    x: int
    a: float
    b: float

    Returns
    -------
    float
        a * np.log(x) + b
    """
    x = np.log(x)
    return a*x + b

all_models['logx_linear'] = logx_linear
model_para['logx_linear'] = [0.378106, 0.046506]
model_para_num['logx_linear'] = 2

def dr_hill_zero_background(x, theta, eta, kappa):
    """dr hill zero background

    Parameters
    ----------
    x: int
    theta: float
    eta: float
    kappa: float

    Returns
    -------
    float
        (theta* x**eta) / (kappa**eta + x**eta)
    """
    return (theta* x**eta) / (kappa**eta + x**eta)

all_models['dr_hill_zero_background'] = dr_hill_zero_background
model_para['dr_hill_zero_background'] = [0.772320, 0.586449, 2.460843]
model_para_num['dr_hill_zero_background'] = 3

def log_power(x, a, b, c):
    """"logistic power

    Parameters
    ----------
    x: int
    a: float
    b: float
    c: float

    Returns
    -------
    float
        a/(1.+(x/np.exp(b))**c)
    """
    return a/(1.+(x/np.exp(b))**c)

all_models['log_power'] = log_power
model_para['log_power'] = [0.77, 2.98, -0.51]
model_para_num['log_power'] = 3

def pow4(x, alpha, a, b, c):
    """pow4

    Parameters
    ----------
    x: int
    alpha: float
    a: float
    b: float
    c: float

    Returns
    -------
    float
        c - (a*x+b)**-alpha
    """
    return c - (a*x+b)**-alpha

all_models['pow4'] = pow4
model_para['pow4'] = [0.1, 200, 0., 0.8]
model_para_num['pow4'] = 4

def mmf(x, alpha, beta, kappa, delta):
    """Morgan-Mercer-Flodin
    http://www.pisces-conservation.com/growthhelp/index.html?morgan_mercer_floden.htm

    Parameters
    ----------
    x: int
    alpha: float
    beta: float
    kappa: float
    delta: float

    Returns
    -------
    float
        alpha - (alpha - beta) / (1. + (kappa * x)**delta)
    """
    return alpha - (alpha - beta) / (1. + (kappa * x)**delta)

all_models['mmf'] = mmf
model_para['mmf'] = [0.7, 0.1, 0.01, 5]
model_para_num['mmf'] = 4

def exp4(x, c, a, b, alpha):
    """exp4

    Parameters
    ----------
    x: int
    c: float
    a: float
    b: float
    alpha: float

    Returns
    -------
    float
        c - np.exp(-a*(x**alpha)+b)
    """
    return c - np.exp(-a*(x**alpha)+b)

all_models['exp4'] = exp4
model_para['exp4'] = [0.7, 0.8, -0.8, 0.3]
model_para_num['exp4'] = 4

def ilog2(x, c, a):
    """ilog2

    Parameters
    ----------
    x: int
    c: float
    a: float

    Returns
    -------
    float
        c - a / np.log(x)
    """
    return c - a / np.log(x)

all_models['ilog2'] = ilog2
model_para['ilog2'] = [0.78, 0.43]
model_para_num['ilog2'] = 2

def weibull(x, alpha, beta, kappa, delta):
    """Weibull model
    http://www.pisces-conservation.com/growthhelp/index.html?morgan_mercer_floden.htm

    Parameters
    ----------
    x: int
    alpha: float
    beta: float
    kappa: float
    delta: float

    Returns
    -------
    float
        alpha - (alpha - beta) * np.exp(-(kappa * x)**delta)
    """
    return alpha - (alpha - beta) * np.exp(-(kappa * x)**delta)

all_models['weibull'] = weibull
model_para['weibull'] = [0.7, 0.1, 0.01, 1]
model_para_num['weibull'] = 4

def janoschek(x, a, beta, k, delta):
    """http://www.pisces-conservation.com/growthhelp/janoschek.htm

    Parameters
    ----------
    x: int
    a: float
    beta: float
    k: float
    delta: float

    Returns
    -------
    float
        a - (a - beta) * np.exp(-k*x**delta)
    """
    return a - (a - beta) * np.exp(-k*x**delta)

all_models['janoschek'] = janoschek
model_para['janoschek'] = [0.73, 0.07, 0.355, 0.46]
model_para_num['janoschek'] = 4
