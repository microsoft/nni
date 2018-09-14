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

'''
Train the network combined by RNN and attention.
'''

import tensorflow as tf

from attention import DotAttention
from rnn import XGRUCell
from util import dropout
from graph_to_tf import graph_to_network


class GAGConfig:
    """The class for model hyper-parameter configuration."""
    def __init__(self):
        self.batch_size = 128

        self.dropout = 0.1

        self.char_vcb_size = 1500
        self.max_char_length = 20
        self.char_embed_dim = 100

        self.max_query_length = 40
        self.max_passage_length = 800

        self.att_is_vanilla = True
        self.att_need_padding = False
        self.att_is_id = False

        self.ptr_dim = 70
        self.learning_rate = 0.1
        self.labelsmoothing = 0.1
        self.num_heads = 1
        self.rnn_units = 256


class GAG:
    """The class for the computation graph based QA model."""
    def __init__(self, cfg, embed, graph):
        self.cfg = cfg
        self.embed = embed
        self.graph = graph

        self.query_word = None
        self.query_mask = None
        self.query_lengths = None
        self.passage_word = None
        self.passage_mask = None
        self.passage_lengths = None
        self.answer_begin = None
        self.answer_end = None
        self.query_char_ids = None
        self.query_char_lengths = None
        self.passage_char_ids = None
        self.passage_char_lengths = None
        self.passage_states = None
        self.query_states = None
        self.query_init = None
        self.begin_prob = None
        self.end_prob = None
        self.loss = None
        self.train_op = None


    def build_net(self, is_training):
        """Build the whole neural network for the QA model."""
        cfg = self.cfg
        with tf.device('/cpu:0'):
            word_embed = tf.get_variable(
                name='word_embed', initializer=self.embed, dtype=tf.float32, trainable=False)
            char_embed = tf.get_variable(name='char_embed',
                                         shape=[cfg.char_vcb_size,
                                                cfg.char_embed_dim],
                                         dtype=tf.float32)

        # [query_length, batch_size]
        self.query_word = tf.placeholder(dtype=tf.int32,
                                         shape=[None, None],
                                         name='query_word')
        self.query_mask = tf.placeholder(dtype=tf.float32,
                                         shape=[None, None],
                                         name='query_mask')
        # [batch_size]
        self.query_lengths = tf.placeholder(
            dtype=tf.int32, shape=[None], name='query_lengths')

        # [passage_length, batch_size]
        self.passage_word = tf.placeholder(
            dtype=tf.int32, shape=[None, None], name='passage_word')
        self.passage_mask = tf.placeholder(
            dtype=tf.float32, shape=[None, None], name='passage_mask')
        # [batch_size]
        self.passage_lengths = tf.placeholder(
            dtype=tf.int32, shape=[None], name='passage_lengths')

        if is_training:
            self.answer_begin = tf.placeholder(
                dtype=tf.int32, shape=[None], name='answer_begin')
            self.answer_end = tf.placeholder(
                dtype=tf.int32, shape=[None], name='answer_end')

        self.query_char_ids = tf.placeholder(dtype=tf.int32,
                                             shape=[
                                                 self.cfg.max_char_length, None, None],
                                             name='query_char_ids')
        # sequence_length, batch_size
        self.query_char_lengths = tf.placeholder(
            dtype=tf.int32, shape=[None, None], name='query_char_lengths')

        self.passage_char_ids = tf.placeholder(dtype=tf.int32,
                                               shape=[
                                                   self.cfg.max_char_length, None, None],
                                               name='passage_char_ids')
        # sequence_length, batch_size
        self.passage_char_lengths = tf.placeholder(dtype=tf.int32,
                                                   shape=[None, None],
                                                   name='passage_char_lengths')

        query_char_states = self.build_char_states(char_embed=char_embed,
                                                   is_training=is_training,
                                                   reuse=False,
                                                   char_ids=self.query_char_ids,
                                                   char_lengths=self.query_char_lengths)

        passage_char_states = self.build_char_states(char_embed=char_embed,
                                                     is_training=is_training,
                                                     reuse=True,
                                                     char_ids=self.passage_char_ids,
                                                     char_lengths=self.passage_char_lengths)

        with tf.variable_scope("encoding") as scope:
            query_states = tf.concat([tf.nn.embedding_lookup(
                word_embed, self.query_word), query_char_states], axis=2)
            scope.reuse_variables()
            passage_states = tf.concat([tf.nn.embedding_lookup(
                word_embed, self.passage_word), passage_char_states], axis=2)
        passage_states = tf.transpose(passage_states, perm=[1, 0, 2])
        query_states = tf.transpose(query_states, perm=[1, 0, 2])
        self.passage_states = passage_states
        self.query_states = query_states

        output, output2 = graph_to_network(passage_states, query_states,
                                           self.passage_lengths, self.query_lengths,
                                           self.graph, self.cfg.dropout,
                                           is_training, num_heads=cfg.num_heads,
                                           rnn_units=cfg.rnn_units)

        passage_att_mask = self.passage_mask
        batch_size_x = tf.shape(self.query_lengths)
        answer_h = tf.zeros(
            tf.concat([batch_size_x, tf.constant([cfg.ptr_dim], dtype=tf.int32)], axis=0))

        answer_context = tf.reduce_mean(output2, axis=1)

        query_init_w = tf.get_variable(
            'query_init_w', shape=[output2.get_shape().as_list()[-1], cfg.ptr_dim])
        self.query_init = query_init_w
        answer_context = tf.matmul(answer_context, query_init_w)

        output = tf.transpose(output, perm=[1, 0, 2])

        with tf.variable_scope('answer_ptr_layer'):
            ptr_att = DotAttention('ptr',
                                   hidden_dim=cfg.ptr_dim,
                                   is_vanilla=self.cfg.att_is_vanilla,
                                   is_identity_transform=self.cfg.att_is_id,
                                   need_padding=self.cfg.att_need_padding)
            answer_pre_compute = ptr_att.get_pre_compute(output)
            ptr_gru = XGRUCell(hidden_dim=cfg.ptr_dim)
            begin_prob, begin_logits = ptr_att.get_prob(output, answer_context, passage_att_mask,
                                                        answer_pre_compute, True)
            att_state = ptr_att.get_att(output, begin_prob)
            (_, answer_h) = ptr_gru.call(inputs=att_state, state=answer_h)
            answer_context = answer_h
            end_prob, end_logits = ptr_att.get_prob(output, answer_context,
                                                    passage_att_mask, answer_pre_compute,
                                                    True)

        self.begin_prob = tf.transpose(begin_prob, perm=[1, 0])
        self.end_prob = tf.transpose(end_prob, perm=[1, 0])
        begin_logits = tf.transpose(begin_logits, perm=[1, 0])
        end_logits = tf.transpose(end_logits, perm=[1, 0])

        if is_training:
            def label_smoothing(inputs, masks, epsilon=0.1):
                """Modify target for label smoothing."""
                epsilon = cfg.labelsmoothing
                num_of_channel = tf.shape(inputs)[-1]  # number of channels
                inputs = tf.cast(inputs, tf.float32)
                return (((1 - epsilon) * inputs) + (epsilon /
                                                    tf.cast(num_of_channel, tf.float32))) * masks
            cost1 = tf.reduce_mean(
                tf.losses.softmax_cross_entropy(label_smoothing(
                    tf.one_hot(self.answer_begin,
                               depth=tf.shape(self.passage_word)[0]),
                    tf.transpose(self.passage_mask, perm=[1, 0])), begin_logits))
            cost2 = tf.reduce_mean(
                tf.losses.softmax_cross_entropy(
                    label_smoothing(tf.one_hot(self.answer_end,
                                               depth=tf.shape(self.passage_word)[0]),
                                    tf.transpose(self.passage_mask, perm=[1, 0])), end_logits))

            reg_ws = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            l2_loss = tf.reduce_sum(reg_ws)
            loss = cost1 + cost2 + l2_loss
            self.loss = loss

            optimizer = tf.train.AdamOptimizer(learning_rate=cfg.learning_rate)
            self.train_op = optimizer.minimize(self.loss)

        return tf.stack([self.begin_prob, self.end_prob])

    def build_char_states(self, char_embed, is_training, reuse, char_ids, char_lengths):
        """Build char embedding network for the QA model."""
        max_char_length = self.cfg.max_char_length

        inputs = dropout(tf.nn.embedding_lookup(char_embed, char_ids),
                         self.cfg.dropout, is_training)
        inputs = tf.reshape(
            inputs, shape=[max_char_length, -1, self.cfg.char_embed_dim])
        char_lengths = tf.reshape(char_lengths, shape=[-1])
        with tf.variable_scope('char_encoding', reuse=reuse):
            cell_fw = XGRUCell(hidden_dim=self.cfg.char_embed_dim)
            cell_bw = XGRUCell(hidden_dim=self.cfg.char_embed_dim)
            _, (left_right, right_left) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                sequence_length=char_lengths,
                inputs=inputs,
                time_major=True,
                dtype=tf.float32
            )

        left_right = tf.reshape(left_right, shape=[-1, self.cfg.char_embed_dim])

        right_left = tf.reshape(right_left, shape=[-1, self.cfg.char_embed_dim])

        states = tf.concat([left_right, right_left], axis=1)
        out_shape = tf.shape(char_ids)[1:3]
        out_shape = tf.concat([out_shape, tf.constant(
            value=[self.cfg.char_embed_dim * 2], dtype=tf.int32)], axis=0)
        return tf.reshape(states, shape=out_shape)
