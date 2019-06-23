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


import random

from .env_vars import trial_env_vars
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
    'mutable_layer',
    'reload_tensorflow_variables'
]


# pylint: disable=unused-argument

if trial_env_vars.NNI_PLATFORM is None:
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

    def mutable_layer():
        raise RuntimeError('Cannot call nni.mutable_layer in this mode')

    def reload_tensorflow_variables(session, tf):
        raise RuntimeError(
            'Cannot call nni.reload_tensorflow_variables in this mode')

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

    def mutable_layer(
            mutable_id,
            mutable_layer_id,
            funcs,
            funcs_args,
            fixed_inputs,
            optional_inputs,
            optional_input_size,
            tf=None,
            mode='general'):
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
        global train_mode
        train_mode = mode
        if mode == 'general':
            mutable_block = _get_param(mutable_id)
            chosen_layer = mutable_block[mutable_layer_id]["chosen_layer"]
            chosen_inputs = mutable_block[mutable_layer_id]["chosen_inputs"]
            real_chosen_inputs = [optional_inputs[input_name]
                                  for input_name in chosen_inputs]
            layer_out = funcs[chosen_layer](
                [fixed_inputs, real_chosen_inputs], **funcs_args[chosen_layer])
        elif mode == 'general-tf':
            assert tf is not None, 'Internal Error: Tensorflow should not be None in oneshot-tf mode'
            name_prefix = "{}_{}".format(mutable_id, mutable_layer_id)
            # store namespace
            if 'name_space' not in globals():
                global name_space
                name_space = dict()
            name_space[mutable_id] = True
            name_space[name_prefix] = dict()
            name_space[name_prefix]['funcs'] = list(funcs)
            name_space[name_prefix]['optional_inputs'] = list(optional_inputs)
            # create tensorflow variables as 1/0 signals used to form subgraph
            if 'tf_variables' not in globals():
                global tf_variables
                tf_variables = dict()
            name_for_optional_inputs = name_prefix + '_optional_inputs'
            name_for_funcs = name_prefix + '_funcs'
            tf_variables[name_prefix] = dict()
            tf_variables[name_prefix]['optional_inputs'] = tf.get_variable(name_for_optional_inputs,
                                                                           [len(optional_inputs)],
                                                                           dtype=tf.bool,
                                                                           trainable=False)
            tf_variables[name_prefix]['funcs'] = tf.get_variable(
                name_for_funcs, [], dtype=tf.int64, trainable=False)

            # get real values using their variable names
            real_optional_inputs_value = [optional_inputs[name]
                                          for name in name_space[name_prefix]['optional_inputs']]
            real_func_value = [funcs[name]
                               for name in name_space[name_prefix]['funcs']]
            real_funcs_args = [funcs_args[name]
                               for name in name_space[name_prefix]['funcs']]
            # build tensorflow graph of geting chosen inputs by masking
            real_chosen_inputs = tf.boolean_mask(
                real_optional_inputs_value, tf_variables[name_prefix]['optional_inputs'])
            # build tensorflow graph of different branches by using tf.case
            branches = dict()
            for func_id in range(len(funcs)):
                func_output = real_func_value[func_id]([fixed_inputs, real_chosen_inputs], **real_funcs_args[func_id])
                branches[tf.equal(tf_variables[name_prefix]['funcs'], func_id)] = lambda: func_output
            layer_out = tf.case(branches, exclusive=True,
                                default=lambda: func_output)
        elif mode == 'oneshot-tf':
            inputs = optional_inputs
            inputs_num = len(inputs)
            # Calculate dropout rate according to the formular r^(1/k), where r is a hyper-parameter and k is the number of inputs
            if inputs_num > 0:
                rate = 0.01 ** (1 / inputs_num)
                noise_shape = [inputs_num] + [1] * inputs[0].get_shape().as_list()
                inputs = tf.nn.dropout(inputs, rate=rate, noise_shape=noise_shape)
            layer_outs = [func([fixed_inputs, inputs], **funcs_args[chosen_layer]) for func in funcs]
            noise_shape = [len(layer_outs)] + [1] * layer_outs[0].get_shape().as_list()
            layer_outs = tf.nn.dropout(layer_outs, rate=rate, noise_shape=noise_shape)
            layer_out = tf.add_n(layer_outs)
        elif mode == 'darts-tf':
            layer_outs = [func([fixed_inputs, inputs], **funcs_args[chosen_layer]) for func in funcs]
            # Create architecture weights for every func(op)
            var_name = "{}_{}_".format(mutable_id, mutable_layer_id, "arch_weights")
            if 'arch_logits_list' not in globals():
                global arch_logits_list
                arch_logits_list = list()
            arch_logits = tf.get_variable(var_name, shape=[len[funcs]], trainable=False)
            arch_logits_list.append(arch_logits)
            arch_weights = tf.nn.softmax(arch_logits)
            layer_out = tf.add_n([arch_weights[idx] * out for idx, out in enumerate(layer_outs)])
        elif mode == 'darts-pytorch':
            raise NotImplementedError()

        return layer_out

    def reload_tensorflow_variables(session, tf=None):
        if train_mode == 'general-tf':
            subgraph_from_tuner = trial.get_next_parameter()
            for mutable_id, mutable_block in subgraph_from_tuner.items():
                if mutable_id not in name_space:
                    continue
                for mutable_layer_id, mutable_layer in mutable_block.items():
                    name_prefix = "{}_{}".format(mutable_id, mutable_layer_id)
                    # extract layer information from the subgraph sampled by tuner
                    chosen_layer = name_space[name_prefix]['funcs'].index(
                        mutable_layer["chosen_layer"])
                    chosen_inputs = [1 if inp in mutable_layer["chosen_inputs"]
                                    else 0 for inp in name_space[name_prefix]['optional_inputs']]
                    # load these information into pre-defined tensorflow variables
                    tf_variables[name_prefix]['funcs'].load(chosen_layer, session)
                    tf_variables[name_prefix]['optional_inputs'].load(
                        chosen_inputs, session)
        elif train_mode == 'darts-tf':
            if 'optimizer' not in globals():
                global arch_logits_list
                global optimizer
                global train_op
                optimizer = tf.MomentumOptimizer(learning_rate=0.025)
                # TODO: Calculate loss
                grads_and_vars = optimizer.compute_gradients(0, arch_logits_list)
                train_op = optimizer.apply_gradients(grads_and_vars)
            session.run(train_op)


    def _get_param(key):
        if trial._params is None:
            trial.get_next_parameter()
        return trial.get_current_parameter(key)
