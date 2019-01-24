import numpy as np
import matplotlib.pyplot as plt
import scipy
import math

all_models = {}
model_defaults = {}
model_para = {}
model_fit = {}
display_name_mapping = {}

def vap(x, a, b, c):
    ''' Vapor pressure model '''
    return np.exp(a+b/x+c*np.log(x))
all_models['vap'] = vap
model_para['vap'] = [-0.622028, -0.470050, 0.042322]
model_defaults['vap'] = {'a': -0.622028, 'c': 0.042322, 'b': -0.470050}

def pow3(x, c, a, alpha):
    return c - a * x**(-alpha)
all_models['pow3'] = pow3
model_para['pow3'] = [0.84, 0.52, 0.01]
model_defaults['pow3'] = {'c': 0.84, 'a': 0.52, 'alpha': 0.01}

def linear(x, a, b):
    return a*x + b
all_models['linear'] =linear

def logx_linear(x, a, b):
    x = np.log(x)
    return a*x + b
all_models['logx_linear'] = logx_linear
model_para['logx_linear'] = [0.378106, 0.046506] 
model_defaults['logx_linear'] = {'a': 0.378106, 'b': 0.046506}

def dr_hill_zero_background(x, theta, eta, kappa):
    return (theta* x**eta) / (kappa**eta + x**eta)
all_models['dr_hill_zero_background'] = dr_hill_zero_background
model_para['dr_hill_zero_background'] = [0.772320, 0.586449, 2.460843]
model_defaults['dr_hill_zero_background'] = {'theta': 0.772320, 'eta': 0.586449, 'kappa': 2.460843}

def log_power(x, a, b, c):
    #logistic power
    return a/(1.+(x/np.exp(b))**c)
all_models['log_power'] = log_power
model_para['log_power'] = [0.77, 2.98, -0.51]
model_defaults['log_power'] = {'a': 0.77, 'c': -0.51, 'b': 2.98}

def pow4(x, alpha, a, b, c):
    return c - (a*x+b)**-alpha
all_models['pow4'] = pow4
model_para['pow4'] = [0.1, 200, 0., 0.8]
model_fit['pow4'] = [0.1, 200, 0., 0.8]
model_defaults['pow4'] = {'alpha': 0.1, 'a':200, 'b':0., 'c': 0.8}

def mmf(x, alpha, beta, kappa, delta):
    return alpha - (alpha - beta) / (1. + (kappa * x)**delta)
all_models['mmf'] = mmf
model_para['mmf'] = [0.7, 0.1, 0.01, 5]
model_fit['mmf'] = [0.7, 0.1, 0.01, 5]
model_defaults['mmf'] = {'alpha': .7, 'kappa': 0.01, 'beta': 0.1, 'delta': 5}

def exp4(x, c, a, b, alpha):
    return c - np.exp(-a*(x**alpha)+b)
all_models['exp4'] = exp4
model_para['exp4'] = [0.7, 0.8, -0.8, 0.3]
model_fit['exp4'] = [0.7, 0.8, -0.8, 0.3]
model_defaults['exp4'] = {'c': 0.7, 'a': 0.8, 'b': -0.8, 'alpha': 0.3}

def ilog2(x, c, a):
    x = 1 + x
    assert(np.all(x > 1))
    return c - a / np.log(x)
all_models['ilog2'] = ilog2
model_para['ilog2'] = [0.78, 0.43]
model_defaults['ilog2'] = {'a': 0.43, 'c': 0.78}

def weibull(x, alpha, beta, kappa,delta):
    return alpha - (alpha - beta) * np.exp(-(kappa * x)**delta)
all_models['weibull'] = weibull
model_para['weibull'] = [0.7, 0.1, 0.01, 1]
model_fit['weibull'] = [0.7, 0.1, 0.01, 1]
model_defaults['weibull'] = {'alpha': 0.7, 'beta': 0.1, 'kappa': 0.01, 'delta': 1}

def janoschek(x, a, beta, k, delta):
    return a - (a - beta) * np.exp(-k*x**delta)
all_models['janoschek'] = janoschek
model_para['janoschek'] = [0.73, 0.07, 0.355, 0.46]
model_fit['janoschek'] = [0.73, 0.07, 0.355, 0.46]
model_defaults['janoschek'] = {'a': 0.73, 'beta': 0.07, 'k': 0.355, 'delta': 0.46}

curve_combination_models = ['pow4', 'mmf', 'exp4', 'weibull', 'janoschek']
