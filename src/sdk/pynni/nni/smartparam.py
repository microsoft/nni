# Copyright (c) Microsoft Corporation. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
# OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ==================================================================================================


import inspect
import math
import random
import numpy as np

from .common import env_args
from . import trial


__all__ = [
    'choice',
    'randint',
    'uniform',
    'quniform',
    'loguniform',
    'qloguniform',
    'normal',
    'qnormal',
    'lognormal',
    'qlognormal',
    'function_choice',
    'get_layer_output',
    'get_choice',
    'get_mask',
    'reload_tf_variable'
]


# pylint: disable=unused-argument

global_layer = dict()

def get_layer_output(layer, layer_name):
    if trial._params is None:
        trial.get_next_parameter()
    input_candidate_names = trial.get_current_parameter(layer_name)['input_candidates']
    layer_choice = trial.get_current_parameter(layer_name)['layer_choice']

    layer = layer[layer_name]
    input_candidate = [layer['input_candidates'][x] for x in input_candidate_names]
    if layer['input_aggregate'] is not None:
        aggregated_output = layer['input_aggregate'](*input_candidate)
        return layer['layer_choice'][layer_choice](layer_name, aggregated_output)
    else:
        return layer['layer_choice'][layer_choice](layer_name, *input_candidate)

def reload_tf_variable(tf, sess):
    '''Get next parameter from tuner and load them into tf variable'''
    global global_layer
    param = trial.get_next_parameter()
    for layer_name, info in global_layer.items():
        mask = [1 if inp in param[layer_name]['input_candidates'] else 0 for inp in info['input_candidates']]
        tf.get_variable(info['mask']).load(mask, sess)
        choice_idx = info['layer_choice'].index(param[layer_name]['layer_choice'])
        tf.get_variable(info['choice']).load(choice_idx, sess)

def get_mask(layer, layer_name, tf):
    '''Create a unique tf variable binary mask for input candidates'''
    mask = tf.get_variable('{}_mask'.format(layer_name),[len(layer[layer_name]['input_candidates'])], dtype=tf.bool, trainable=False)
    layer[layer_name]['mask'] = mask.name
    global global_layer
    global_layer.update(layer)
    return mask

def get_choice(layer, layer_name, tf):
    '''Create a unique tf scalar variable for layer choice'''
    choice = tf.get_variable('{}_choice'.format(layer_name), [], dtype=tf.int64, trainable=False)
    layer[layer_name]['choice'] = choice.name
    global global_layer
    global_layer.update(layer)
    return choice

if env_args.platform is None:
    def choice(*options, name=None):
        return random.choice(options)

    def randint(upper, name=None):
        return random.randrange(upper)

    def uniform(low, high, name=None):
        return random.uniform(low, high)

    def quniform(low, high, q, name=None):
        assert high > low, 'Upper bound must be larger than lower bound'
        return round(random.uniform(low, high) / q) * q

    def loguniform(low, high, name=None):
        assert low > 0, 'Lower bound must be positive'
        return np.exp(random.uniform(np.log(low), np.log(high)))

    def qloguniform(low, high, q, name=None):
        return round(loguniform(low, high) / q) * q

    def normal(mu, sigma, name=None):
        return random.gauss(mu, sigma)

    def qnormal(mu, sigma, q, name=None):
        return round(random.gauss(mu, sigma) / q) * q

    def lognormal(mu, sigma, name=None):
        return np.exp(random.gauss(mu, sigma))

    def qlognormal(mu, sigma, q, name=None):
        return round(lognormal(mu, sigma) / q) * q

    def function_choice(*funcs, name=None):
        return random.choice(funcs)()

else:

    def choice(options, name=None, key=None):
        return options[_get_param(key)]

    def randint(upper, name=None, key=None):
        return _get_param(key)

    def uniform(low, high, name=None, key=None):
        return _get_param(key)

    def quniform(low, high, q, name=None, key=None):
        return _get_param(key)

    def loguniform(low, high, name=None, key=None):
        return _get_param(key)

    def qloguniform(low, high, q, name=None, key=None):
        return _get_param(key)

    def normal(mu, sigma, name=None, key=None):
        return _get_param(key)

    def qnormal(mu, sigma, q, name=None, key=None):
        return _get_param(key)

    def lognormal(mu, sigma, name=None, key=None):
        return _get_param(key)

    def qlognormal(mu, sigma, q, name=None, key=None):
        return _get_param(key)

    def function_choice(funcs, name=None, key=None):
        return funcs[_get_param(key)]()

    def _get_param(key):
        if trial._params is None:
            trial.get_next_parameter()
        return trial.get_current_parameter(key)
