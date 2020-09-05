# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge,
# to any person obtaining a copy of this software and associated
# documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import RNNCell


class GRU:
    '''
    GRU class.
    '''
    def __init__(self, name, input_dim, hidden_dim):
        self.name = '/'.join([name, 'gru'])
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.w_matrix = None
        self.U = None
        self.bias = None

    def define_params(self):
        '''
        Define parameters.
        '''
        input_dim = self.input_dim
        hidden_dim = self.hidden_dim
        prefix = self.name
        self.w_matrix = tf.Variable(tf.random_normal([input_dim, 3 * hidden_dim], stddev=0.1),
                                    name='/'.join([prefix, 'W']))
        self.U = tf.Variable(tf.random_normal([hidden_dim, 3 * hidden_dim], stddev=0.1),
                             name='/'.join([prefix, 'U']))
        self.bias = tf.Variable(tf.random_normal([1, 3 * hidden_dim], stddev=0.1),
                                name='/'.join([prefix, 'b']))
        return self

    def build(self, x, h, mask=None):
        '''
        Build the GRU cell.
        '''
        xw = tf.split(tf.matmul(x, self.w_matrix) + self.bias, 3, 1)
        hu = tf.split(tf.matmul(h, self.U), 3, 1)
        r = tf.sigmoid(xw[0] + hu[0])
        z = tf.sigmoid(xw[1] + hu[1])
        h1 = tf.tanh(xw[2] + r * hu[2])
        next_h = h1 * (1 - z) + h * z
        if mask is not None:
            next_h = next_h * mask + h * (1 - mask)
        return next_h

    def build_sequence(self, xs, masks, init, is_left_to_right):
        '''
        Build GRU sequence.
        '''
        states = []
        last = init
        if is_left_to_right:
            for i, xs_i in enumerate(xs):
                h = self.build(xs_i, last, masks[i])
                states.append(h)
                last = h
        else:
            for i in range(len(xs) - 1, -1, -1):
                h = self.build(xs[i], last, masks[i])
                states.insert(0, h)
                last = h
        return states


class XGRUCell(RNNCell):

    def __init__(self, hidden_dim, reuse=None):
        super(XGRUCell, self).__init__(self, _reuse=reuse)
        self._num_units = hidden_dim
        self._activation = tf.tanh

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):

        input_dim = inputs.get_shape()[-1]
        assert input_dim is not None, "input dimension must be defined"
        W = tf.get_variable(
            name="W", shape=[input_dim, 3 * self._num_units], dtype=tf.float32)
        U = tf.get_variable(
            name='U', shape=[self._num_units, 3 * self._num_units], dtype=tf.float32)
        b = tf.get_variable(
            name='b', shape=[1, 3 * self._num_units], dtype=tf.float32)

        xw = tf.split(tf.matmul(inputs, W) + b, 3, 1)
        hu = tf.split(tf.matmul(state, U), 3, 1)
        r = tf.sigmoid(xw[0] + hu[0])
        z = tf.sigmoid(xw[1] + hu[1])
        h1 = self._activation(xw[2] + r * hu[2])
        next_h = h1 * (1 - z) + state * z
        return next_h, next_h
