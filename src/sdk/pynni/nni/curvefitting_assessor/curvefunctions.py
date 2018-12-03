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

# curve function constant
import numpy as np

# all the models that we considered at some point
all_models = {}
model_defaults = {}

def vap(x, a, b, c):
    ''' Vapor pressure model '''
    return np.exp(a+b/x+c*np.log(x))
all_models['vap'] = vap
model_defaults['vap'] = {'a': -0.622028, 'c': 0.042322, 'b': -0.470050}

def pow3(x, c, a, alpha):
    return c - a * x**(-alpha)
all_models['pow3'] = pow3
model_defaults['pow3'] = {'c': 0.84, 'a': 0.52, 'alpha': 0.01}

def linear(x, a, b):
    return a*x + b
all_models['linear'] =linear

def logx_linear(x, a, b):
    x = np.log(x)
    return a*x + b
all_models['logx_linear'] = logx_linear
model_defaults['logx_linear'] = {'a': 0.378106, 'b': 0.046506}

def loglog_linear(x, a, b):
    x = np.log(x)
    return np.log(a*x + b)
all_models['loglog_linear'] = loglog_linear

def dr_hill_zero_background(x, theta, eta, kappa):
    return (theta* x**eta) / (kappa**eta + x**eta)
all_models['dr_hill_zero_background'] = dr_hill_zero_background
model_defaults['dr_hill_zero_background'] = {'theta': 0.772320, 'eta': 0.586449, 'kappa': 2.460843}

def log_power(x, a, b, c):
    #logistic power
    return a/(1.+(x/np.exp(b))**c)
all_models['log_power'] = log_power
model_defaults['log_power'] = {'a': 0.77, 'c': -0.51, 'b': 2.98}

def pow4(x, c, a, b, alpha):
    return c - (a*x+b)**-alpha
all_models['pow4'] = pow4
model_defaults['pow4'] = {'alpha': 0.1, 'a':200, 'b':0., 'c': 0.8}

def mmf(x, alpha, beta, kappa, delta):
    '''
    Morgan-Mercer-Flodin

    http://www.pisces-conservation.com/growthhelp/index.html?morgan_mercer_floden.htm

    alpha: upper asymptote
    kappa: growth rate
    beta: initial value
    delta: controls the point of inflection
    '''
    # print (alpha, beta, kappa, delta)
    return alpha - (alpha - beta) / (1. + (kappa * x)**delta)
all_models['mmf'] = mmf
model_defaults['mmf'] = {'alpha': .7, 'kappa': 0.01, 'beta': 0.1, 'delta': 5}

def exp4(x, c, a, b, alpha):
    return c - np.exp(-a*(x**alpha)+b)
all_models['exp4'] = exp4
model_defaults['exp4'] = {'c': 0.7, 'a': 0.8, 'b': -0.8, 'alpha': 0.3}

def ilog2(x, c, a):
    x = 1 + x
    assert(np.all(x > 1))
    return c - a / np.log(x)
all_models['ilog2'] = ilog2
model_defaults['ilog2'] = {'a': 0.43, 'c': 0.78}

def weibull(x, alpha, beta, kappa,delta):
    '''
    Weibull model

    http://www.pisces-conservation.com/growthhelp/index.html?morgan_mercer_floden.htm

    alpha: upper asymptote
    beta: lower asymptote
    k: growth rate
    delta: controls the x-orginate for the point of inflection
    '''
    return alpha - (alpha - beta) * np.exp(-(kappa * x)**delta)
all_models['weibull'] = weibull
model_defaults['weibull'] = {'alpha': 0.7, 'beta': 0.1, 'kappa': 0.01, 'delta': 1}

def janoschek(x, a, beta, k, delta):
    '''
    http://www.pisces-conservation.com/growthhelp/janoschek.htm
    ''' 
    return a - (a - beta) * np.exp(-k*x**delta)
all_models['janoschek'] = janoschek
model_defaults['janoschek'] = {'a': 0.73, 'beta': 0.07, 'k': 0.355, 'delta': 0.46}

curve_combination_models = ['vap', 'pow3', 'loglog_linear', 'dr_hill_zero_background', 'log_power', 'pow4', 'mmf', 'exp4', 'ilog2', 'weibull', 'janoschek']
