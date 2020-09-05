# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
util functions
"""

import os
import random
import multiprocessing
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from gym.spaces import Discrete, Box, MultiDiscrete

def set_global_seeds(i):
    """set global seeds"""
    rank = 0
    myseed = i  + 1000 * rank if i is not None else None
    tf.set_random_seed(myseed)
    np.random.seed(myseed)
    random.seed(myseed)

def batch_to_seq(h, nbatch, nsteps, flat=False):
    """convert from batch to sequence"""
    if flat:
        h = tf.reshape(h, [nbatch, nsteps])
    else:
        h = tf.reshape(h, [nbatch, nsteps, -1])
    return [tf.squeeze(v, [1]) for v in tf.split(axis=1, num_or_size_splits=nsteps, value=h)]

def seq_to_batch(h, flat=False):
    """convert from sequence to batch"""
    shape = h[0].get_shape().as_list()
    if not flat:
        assert len(shape) > 1
        nh = h[0].get_shape()[-1].value
        return tf.reshape(tf.concat(axis=1, values=h), [-1, nh])
    else:
        return tf.reshape(tf.stack(values=h, axis=1), [-1])

def lstm(xs, ms, s, scope, nh, init_scale=1.0):
    """lstm cell"""
    _, nin = [v.value for v in xs[0].get_shape()] # the first is nbatch
    with tf.variable_scope(scope):
        wx = tf.get_variable("wx", [nin, nh*4], initializer=ortho_init(init_scale))
        wh = tf.get_variable("wh", [nh, nh*4], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh*4], initializer=tf.constant_initializer(0.0))

    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
    for idx, (x, m) in enumerate(zip(xs, ms)):
        c = c*(1-m)
        h = h*(1-m)
        z = tf.matmul(x, wx) + tf.matmul(h, wh) + b
        i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)
        c = f*c + i*u
        h = o*tf.tanh(c)
        xs[idx] = h
    s = tf.concat(axis=1, values=[c, h])
    return xs, s

def lstm_model(nlstm=128, layer_norm=False):
    """
    Builds LSTM (Long-Short Term Memory) network to be used in a policy.
    Note that the resulting function returns not only the output of the LSTM
    (i.e. hidden state of lstm for each step in the sequence), but also a dictionary
    with auxiliary tensors to be set as policy attributes.

    Specifically,
        S is a placeholder to feed current state (LSTM state has to be managed outside policy)
        M is a placeholder for the mask (used to mask out observations after the end of the episode, but can be used for other purposes too)
        initial_state is a numpy array containing initial lstm state (usually zeros)
        state is the output LSTM state (to be fed into S at the next call)


    An example of usage of lstm-based policy can be found here: common/tests/test_doc_examples.py/test_lstm_example

    Parameters
    ----------
    nlstm : int
        LSTM hidden state size
    layer_norm : bool
        if True, layer-normalized version of LSTM is used

    Returns
    -------
    function that builds LSTM with a given input tensor / placeholder
    """

    def network_fn(X, nenv=1, obs_size=-1):
        with tf.variable_scope("emb", reuse=tf.AUTO_REUSE):
            w_emb = tf.get_variable("w_emb", [obs_size+1, 32])
            X = tf.nn.embedding_lookup(w_emb, X)

        nbatch = X.shape[0]
        nsteps = nbatch // nenv

        h = tf.layers.flatten(X)

        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, 2*nlstm]) #states

        xs = batch_to_seq(h, nenv, nsteps)
        ms = batch_to_seq(M, nenv, nsteps)

        assert not layer_norm
        h5, snew = lstm(xs, ms, S, scope='lstm', nh=nlstm)

        h = seq_to_batch(h5)
        initial_state = np.zeros(S.shape.as_list(), dtype=float)

        return h, {'S':S, 'M':M, 'state':snew, 'initial_state':initial_state}

    return network_fn

def ortho_init(scale=1.0):
    """init approach"""
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init

def fc(x, scope, nh, *, init_scale=1.0, init_bias=0.0):
    """fully connected op"""
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias))
        return tf.matmul(x, w)+b

def _check_shape(placeholder_shape, data_shape):
    """
    check if two shapes are compatible (i.e. differ only by dimensions of size 1, or by the batch dimension)
    """

    return True

# ================================================================
# Shape adjustment for feeding into tf placeholders
# ================================================================
def adjust_shape(placeholder, data):
    """
    adjust shape of the data to the shape of the placeholder if possible.
    If shape is incompatible, AssertionError is thrown

    Parameters
    ----------
    placeholder
        tensorflow input placeholder
    data
        input data to be (potentially) reshaped to be fed into placeholder

    Returns
    -------
    reshaped data
    """
    if not isinstance(data, np.ndarray) and not isinstance(data, list):
        return data
    if isinstance(data, list):
        data = np.array(data)

    placeholder_shape = [x or -1 for x in placeholder.shape.as_list()]

    assert _check_shape(placeholder_shape, data.shape), \
        'Shape of data {} is not compatible with shape of the placeholder {}'.format(data.shape, placeholder_shape)

    return np.reshape(data, placeholder_shape)

# ================================================================
# Global session
# ================================================================

def get_session(config=None):
    """Get default session or create one with a given config"""
    sess = tf.get_default_session()
    if sess is None:
        sess = make_session(config=config, make_default=True)
    return sess

def make_session(config=None, num_cpu=None, make_default=False, graph=None):
    """Returns a session that will use <num_cpu> CPU's only"""
    if num_cpu is None:
        num_cpu = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))
    if config is None:
        config = tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=num_cpu,
            intra_op_parallelism_threads=num_cpu)
        config.gpu_options.allow_growth = True

    if make_default:
        return tf.InteractiveSession(config=config, graph=graph)
    else:
        return tf.Session(config=config, graph=graph)

ALREADY_INITIALIZED = set()

def initialize():
    """Initialize all the uninitialized variables in the global scope."""
    new_variables = set(tf.global_variables()) - ALREADY_INITIALIZED
    get_session().run(tf.variables_initializer(new_variables))

    ALREADY_INITIALIZED.update(new_variables)

def observation_placeholder(ob_space, batch_size=None, name='Ob'):
    """
    Create placeholder to feed observations into of the size appropriate to the observation space

    Parameters
    ----------
    ob_space : gym.Space
        observation space
    batch_size : int
        size of the batch to be fed into input. Can be left None in most cases.
    name : str
        name of the placeholder

    Returns
    -------
    tensorflow placeholder tensor
    """

    assert isinstance(ob_space, (Discrete, Box, MultiDiscrete)), \
        'Can only deal with Discrete and Box observation spaces for now'

    dtype = ob_space.dtype
    if dtype == np.int8:
        dtype = np.uint8

    return tf.placeholder(shape=(batch_size,) + ob_space.shape, dtype=dtype, name=name)

def explained_variance(ypred, y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary == 0 else 1 - np.var(y-ypred)/vary
