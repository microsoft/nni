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

import math

import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import RNNCell


def _get_variable(variable_dict, name, shape, initializer=None, dtype=tf.float32):
    if name not in variable_dict:
        variable_dict[name] = tf.get_variable(
            name=name, shape=shape, initializer=initializer, dtype=dtype)
    return variable_dict[name]


class DotAttention:
    '''
    DotAttention
    '''

    def __init__(self, name,
                 hidden_dim,
                 is_vanilla=True,
                 is_identity_transform=False,
                 need_padding=False):
        self._name = '/'.join([name, 'dot_att'])
        self._hidden_dim = hidden_dim
        self._is_identity_transform = is_identity_transform
        self._need_padding = need_padding
        self._is_vanilla = is_vanilla
        self._var = {}

    @property
    def is_identity_transform(self):
        return self._is_identity_transform

    @property
    def is_vanilla(self):
        return self._is_vanilla

    @property
    def need_padding(self):
        return self._need_padding

    @property
    def hidden_dim(self):
        return self._hidden_dim

    @property
    def name(self):
        return self._name

    @property
    def var(self):
        return self._var

    def _get_var(self, name, shape, initializer=None):
        with tf.variable_scope(self.name):
            return _get_variable(self.var, name, shape, initializer)

    def _define_params(self, src_dim, tgt_dim):
        hidden_dim = self.hidden_dim
        self._get_var('W', [src_dim, hidden_dim])
        if not self.is_vanilla:
            self._get_var('V', [src_dim, hidden_dim])
            if self.need_padding:
                self._get_var('V_s', [src_dim, src_dim])
                self._get_var('V_t', [tgt_dim, tgt_dim])
            if not self.is_identity_transform:
                self._get_var('T', [tgt_dim, src_dim])
        self._get_var('U', [tgt_dim, hidden_dim])
        self._get_var('b', [1, hidden_dim])
        self._get_var('v', [hidden_dim, 1])

    def get_pre_compute(self, s):
        '''
        :param s: [src_sequence, batch_size, src_dim]
        :return: [src_sequence, batch_size. hidden_dim]
        '''
        hidden_dim = self.hidden_dim
        src_dim = s.get_shape().as_list()[-1]
        assert src_dim is not None, 'src dim must be defined'
        W = self._get_var('W', shape=[src_dim, hidden_dim])
        b = self._get_var('b', shape=[1, hidden_dim])
        return tf.tensordot(s, W, [[2], [0]]) + b

    def get_prob(self, src, tgt, mask, pre_compute, return_logits=False):
        '''
        :param s: [src_sequence_length, batch_size, src_dim]
        :param h: [batch_size, tgt_dim] or [tgt_sequence_length, batch_size, tgt_dim]
        :param mask: [src_sequence_length, batch_size]\
             or [tgt_sequence_length, src_sequence_length, batch_sizse]
        :param pre_compute: [src_sequence_length, batch_size, hidden_dim]
        :return: [src_sequence_length, batch_size]\
             or [tgt_sequence_length, src_sequence_length, batch_size]
        '''
        s_shape = src.get_shape().as_list()
        h_shape = tgt.get_shape().as_list()
        src_dim = s_shape[-1]
        tgt_dim = h_shape[-1]
        assert src_dim is not None, 'src dimension must be defined'
        assert tgt_dim is not None, 'tgt dimension must be defined'

        self._define_params(src_dim, tgt_dim)

        if len(h_shape) == 2:
            tgt = tf.expand_dims(tgt, 0)
        if pre_compute is None:
            pre_compute = self.get_pre_compute(src)

        buf0 = pre_compute
        buf1 = tf.tensordot(tgt, self.var['U'], axes=[[2], [0]])
        buf2 = tf.tanh(tf.expand_dims(buf0, 0) + tf.expand_dims(buf1, 1))

        if not self.is_vanilla:
            xh1 = tgt
            xh2 = tgt
            s1 = src
            if self.need_padding:
                xh1 = tf.tensordot(xh1, self.var['V_t'], 1)
                xh2 = tf.tensordot(xh2, self.var['S_t'], 1)
                s1 = tf.tensordot(s1, self.var['V_s'], 1)
            if not self.is_identity_transform:
                xh1 = tf.tensordot(xh1, self.var['T'], 1)
                xh2 = tf.tensordot(xh2, self.var['T'], 1)
            buf3 = tf.expand_dims(s1, 0) * tf.expand_dims(xh1, 1)
            buf3 = tf.tanh(tf.tensordot(buf3, self.var['V'], axes=[[3], [0]]))
            buf = tf.reshape(tf.tanh(buf2 + buf3), shape=tf.shape(buf3))
        else:
            buf = buf2
        v = self.var['v']
        e = tf.tensordot(buf, v, [[3], [0]])
        e = tf.squeeze(e, axis=[3])
        tmp = tf.reshape(e + (mask - 1) * 10000.0, shape=tf.shape(e))
        prob = tf.nn.softmax(tmp, 1)
        if len(h_shape) == 2:
            prob = tf.squeeze(prob, axis=[0])
            tmp = tf.squeeze(tmp, axis=[0])
        if return_logits:
            return prob, tmp
        return prob

    def get_att(self, s, prob):
        '''
        :param s: [src_sequence_length, batch_size, src_dim]
        :param prob: [src_sequence_length, batch_size]\
            or [tgt_sequence_length, src_sequence_length, batch_size]
        :return: [batch_size, src_dim] or [tgt_sequence_length, batch_size, src_dim]
        '''
        buf = s * tf.expand_dims(prob, axis=-1)
        att = tf.reduce_sum(buf, axis=-3)
        return att
