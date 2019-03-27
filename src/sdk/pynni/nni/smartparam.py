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

def get_layer_output(layer, layer_name, tf=None):
    current_layer = layer[layer_name]
    # Get inputs
    if tf is None:
        if trial._params is None:
            trial.get_next_parameter()
        input_candidate_names = trial.get_current_parameter(layer_name)['input_candidates']
        layer_choice = trial.get_current_parameter(layer_name)['layer_choice']
        layer_inputs = [current_layer['input_candidates'][x] for x in input_candidate_names]
    else:
        layer_mask = get_mask(layer, layer_name, tf)
        layer_inputs = tf.boolean_mask(current_layer['input_candidates'], layer_mask)
    # Invoke input_aggregate function if it's not None
    if current_layer['input_aggregate'] is not None:
        layer_inputs = current_layer['input_aggregate'](layer_inputs)
    # Get output
    if tf is None:
        layer_out = current_layer['layer_choice'][layer_choice](layer_name, layer_inputs)
    else:
        layer_choice = get_choice(layer, layer_name, tf)
        layer_branches = {}
        for idx in range(len(current_layer['layer_choice'])):
            layer_y = current_layer['layer_choice'][idx](layer_name, layer_inputs)
            layer_branches[tf.equal(layer_choice, idx)] = lambda: layer_y
        layer_out = tf.case(layer_branches, exclusive=True, default=lambda: layer_y)
    # Invoke post_process_outputs function if it's not None
    if current_layer['post_process_outputs'] is not None:
        layer_out = current_layer['post_process_outputs'](layer_name, layer_out, layer_inputs)
    
    return layer_out


def reload_tf_variable(tf=None, session=None):
    '''Get next parameter from tuner and load them into tf variable'''
    if tf is not None:
        assert session is not None, "Tensorflow session should be provided"
        global global_layer
        trial.get_next_parameter()
        param = trial._params['parameters']
        for layer_name, info in global_layer.items():
            mask = [1 if inp in param[layer_name]['input_candidates']
                    else 0 for inp in info['input_candidates_str']]
            info['mask'].load(mask, session)
            choice_idx = info['layer_choice_str'].index(
                param[layer_name]['layer_choice'])
            info['choice'].load(choice_idx, session)


def get_mask(layer, layer_name, tf):
    '''Create a unique tf variable binary mask for input candidates'''
    current_layer = layer[layer_name]
    current_layer['mask'] = tf.get_variable('{}_mask'.format(layer_name), [len(
        current_layer['input_candidates'])], dtype=tf.bool, trainable=False)
    global global_layer
    global_layer.update(layer)
    return current_layer['mask']


def get_choice(layer, layer_name, tf):
    '''Create a unique tf scalar variable for layer choice'''
    current_layer = layer[layer_name]
    current_layer['choice'] = tf.get_variable('{}_choice'.format(
        layer_name), [], dtype=tf.int64, trainable=False)
    global global_layer
    global_layer.update(layer)
    return current_layer['choice']


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
