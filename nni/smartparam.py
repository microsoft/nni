# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

from .runtime.env_vars import trial_env_vars
from . import trial
from . import parameter_expressions as param_exp
from .common.nas_utils import classic_mode, enas_mode, oneshot_mode, darts_mode


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
    'mutable_layer'
]


if trial_env_vars.NNI_PLATFORM is None:
    def choice(*options, name=None):
        return param_exp.choice(options, np.random.RandomState())

    def randint(lower, upper, name=None):
        return param_exp.randint(lower, upper, np.random.RandomState())

    def uniform(low, high, name=None):
        return param_exp.uniform(low, high, np.random.RandomState())

    def quniform(low, high, q, name=None):
        assert high > low, 'Upper bound must be larger than lower bound'
        return param_exp.quniform(low, high, q, np.random.RandomState())

    def loguniform(low, high, name=None):
        assert low > 0, 'Lower bound must be positive'
        return param_exp.loguniform(low, high, np.random.RandomState())

    def qloguniform(low, high, q, name=None):
        return param_exp.qloguniform(low, high, q, np.random.RandomState())

    def normal(mu, sigma, name=None):
        return param_exp.normal(mu, sigma, np.random.RandomState())

    def qnormal(mu, sigma, q, name=None):
        return param_exp.qnormal(mu, sigma, q, np.random.RandomState())

    def lognormal(mu, sigma, name=None):
        return param_exp.lognormal(mu, sigma, np.random.RandomState())

    def qlognormal(mu, sigma, q, name=None):
        return param_exp.qlognormal(mu, sigma, q, np.random.RandomState())

    def function_choice(*funcs, name=None):
        return param_exp.choice(funcs, np.random.RandomState())()

    def mutable_layer():
        raise RuntimeError('Cannot call nni.mutable_layer in this mode')

else:

    def choice(options, name=None, key=None):
        return options[_get_param(key)]

    def randint(lower, upper, name=None, key=None):
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

    def mutable_layer(
            mutable_id,
            mutable_layer_id,
            funcs,
            funcs_args,
            fixed_inputs,
            optional_inputs,
            optional_input_size,
            mode='classic_mode',
            tf=None):
        '''execute the chosen function and inputs.
        Below is an example of chosen function and inputs:
        {
            "mutable_id": {
                "mutable_layer_id": {
                    "chosen_layer": "pool",
                    "chosen_inputs": ["out1", "out3"]
                }
            }
        }
        Parameters:
        ---------------
        mutable_id: the name of this mutable_layer block (which could have multiple mutable layers)
        mutable_layer_id: the name of a mutable layer in this block
        funcs: dict of function calls
        funcs_args:
        fixed_inputs:
        optional_inputs: dict of optional inputs
        optional_input_size: number of candidate inputs to be chosen
        tf: tensorflow module
        '''
        args = (mutable_id, mutable_layer_id, funcs, funcs_args, fixed_inputs, optional_inputs, optional_input_size)
        if mode == 'classic_mode':
            return classic_mode(*args)
        assert tf is not None, 'Internal Error: Tensorflow should not be None in modes other than classic_mode'
        if mode == 'enas_mode':
            return enas_mode(*args, tf)
        if mode == 'oneshot_mode':
            return oneshot_mode(*args, tf)
        if mode == 'darts_mode':
            return darts_mode(*args, tf)
        raise RuntimeError('Unrecognized mode: %s' % mode)

    def _get_param(key):
        if trial.get_current_parameter() is None:
            trial.get_next_parameter()
        return trial.get_current_parameter(key)
