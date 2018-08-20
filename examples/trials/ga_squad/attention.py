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


def batch_linear_layer(matrix_a, matrix_b):
    '''
    shape of matrix_a is [*, batch, dima]
    shape of matrix_b is [batch, dima, dimb]
    result is [*, batch, dimb]
    for each batch, do matrix_a linear op to last dim
    '''
    matrix_a = tf.expand_dims(matrix_a, -1)
    while len(list(matrix_b.shape)) < len(list(matrix_a.shape)):
        matrix_b = tf.expand_dims(matrix_b, 0)
    return tf.reduce_sum(matrix_a * matrix_b, -2)


def split_last_dim(x, factor):
    shape = tf.shape(x)
    last_dim = int(x.shape[-1])
    assert last_dim % factor == 0, \
        "last dim isn't divisible by factor {%d} {%d}" % (last_dim, factor)
    new_shape = tf.concat(
        [shape[:-1], tf.constant([factor, last_dim // factor])], axis=0)
    return tf.reshape(x, new_shape)


def merge_last2_dim(x):
    shape = tf.shape(x)
    last_dim = int(x.shape[-1]) * int(x.shape[-2])
    new_shape = tf.concat([shape[:-2], tf.constant([last_dim])], axis=0)
    return tf.reshape(x, new_shape)


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


class MultiHeadAttention:
    '''
    MultiHeadAttention.
    '''
    def __init__(self, name, hidden_dim, head, add=True, dot=True, divide=True):
        self._name = '/'.join([name, 'dot_att'])
        self._head = head
        self._head_dim = hidden_dim // head
        self._hidden_dim = self._head_dim * head
        self._add = add
        self._dot = dot
        assert add or dot, "you must at least choose one between add and dot"
        self._div = 1.0
        if divide:
            self._div = math.sqrt(self._head_dim)
        self._var = {}

    @property
    def hidden_dim(self):
        return self._head_dim * self._head

    @property
    def name(self):
        return self._name

    @property
    def var(self):
        return self._var

    def _get_var(self, name, shape, initializer=None):
        with tf.variable_scope(self.name):
            return _get_variable(self.var, name, shape, initializer)

    def _define_params(self, tgt_dim):
        self._get_var('tgt_project', [tgt_dim, self._hidden_dim])
        self._get_var('tgt_bias', [1, self._hidden_dim])
        self._get_var('v', [self._head, self._head_dim, 1])

    def get_pre_compute(self, src):
        s_shape = src.get_shape().as_list()
        src_dim = s_shape[-1]
        src_project = self._get_var('src_project', [src_dim, self._hidden_dim])
        src_bias = self._get_var('src_bias', [1, self._hidden_dim])
        src = split_last_dim(tf.tensordot(src, src_project,
                                          [[2], [0]]) + src_bias, self._head)
        return src

    def get_prob(self, src, tgt, mask, pre_compute):
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
        print('src tgt dim: ', src_dim, tgt_dim)
        assert src_dim is not None, 'src dimension must be defined'
        assert tgt_dim is not None, 'tgt dimension must be defined'

        self._define_params(tgt_dim)

        if len(h_shape) == 2:
            tgt = tf.expand_dims(tgt, 0)

        tgt_project = self._var['tgt_project']
        tgt_bias = self._var['tgt_bias']

        if pre_compute is None:
            pre_compute = self.get_pre_compute(src)

        src = pre_compute
        tgt = split_last_dim(tf.tensordot(tgt, tgt_project,
                                          [[2], [0]]) + tgt_bias, self._head)

        add_attention = 0
        dot_attention = 0
        if self._add:
            buf = tf.tanh(tf.expand_dims(src, 0) + tf.expand_dims(tgt, 1))
            v = self.var['v']
            add_attention = tf.squeeze(batch_linear_layer(buf, v), -1)
        if self._dot:
            dot_attention = tf.reduce_sum(tf.expand_dims(
                src, 0) * tf.expand_dims(tgt, 1), -1)
            dot_attention /= self._div

        attention = add_attention + dot_attention
        mask = tf.expand_dims(mask, -1)
        logits = attention + (mask - 1) * 10000.0
        prob = tf.nn.softmax(logits, 1)
        if len(h_shape) == 2:
            prob = tf.squeeze(prob, axis=[0])
        return prob

    def map_target(self, tgt):
        tgt_project = self._var['tgt_project']
        tgt_bias = self._var['tgt_bias']
        tgt = tf.tensordot(tgt, tgt_project, [[1], [0]]) + tgt_bias
        return tgt

    def get_att(self, src, prob):
        '''
        :param s: [src_sequence_length, batch_size, head, head_dim]
        :param prob: [src_sequence_length, batch_size, head]\
             or [tgt_sequence_length, src_sequence_length, batch_size, head]
        :return: [batch_size, src_dim] or [tgt_sequence_length, batch_size, src_dim]
        '''
        buf = merge_last2_dim(tf.reduce_sum(
            src * tf.expand_dims(prob, axis=-1), axis=-4))
        return buf


class DotAttentionWrapper(RNNCell):
    '''
    A wrapper for DotAttention or MultiHeadAttention.
    '''

    def __init__(self, cell, attention,
                 src, mask, is_gated,
                 reuse=None, dropout=None,
                 keep_input=True, map_target=False):
        super().__init__(self, _reuse=reuse)
        assert isinstance(attention, (DotAttention, MultiHeadAttention)), \
            'type of attention is not supported'
        assert isinstance(cell, RNNCell), 'type of cell must be RNNCell'
        self._attention = attention
        self._src = src
        self._mask = mask
        self._pre_computed = None
        self._is_gated = is_gated
        self._cell = cell
        self._dropout = dropout
        self._keep_input = keep_input
        self._map_target = map_target

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def call(self, inputs, state):
        if self._pre_computed is None:
            self._pre_computed = self._attention.get_pre_compute(self._src)
        att_prob = self._attention.get_prob(
            src=self._src,
            tgt=tf.concat([inputs, state], axis=1),
            mask=self._mask,
            pre_compute=self._pre_computed)
        if isinstance(self._attention, DotAttention):
            att = self._attention.get_att(self._src, att_prob)
        else:
            att = self._attention.get_att(self._pre_computed, att_prob)
        x_list = [att]
        if self._keep_input:
            x_list.append(inputs)
            if inputs.shape[1] == att.shape[1]:
                x_list.append(inputs - att)
                x_list.append(inputs * att)
        if self._map_target and isinstance(self._attention, MultiHeadAttention):
            tgt = self._attention.map_target(
                tf.concat([inputs, state], axis=1))
            x_list += [tgt, att-tgt, att*tgt]

        x = tf.concat(x_list, axis=1)
        dim = x.get_shape().as_list()[1]
        assert dim is not None, 'dim must be defined'
        if self._is_gated:
            g = tf.get_variable('att_gate',
                                shape=[dim, dim],
                                dtype=tf.float32,
                                initializer=None)
            bias_g = tf.get_variable(
                'bias_gate', shape=[1, dim], dtype=tf.float32)
            gate = tf.sigmoid(tf.matmul(x, g) + bias_g)
            x = x * gate
        if self._dropout is not None:
            x = self._dropout(x)
        return self._cell.call(x, state)
