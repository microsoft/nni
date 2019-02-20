from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import time

import numpy as np
import tensorflow as tf

from src.utils import get_train_ops
from src.common_ops import stack_lstm

from tensorflow.python.training import moving_averages

class PTBEnasController(object):
  def __init__(self,
               rhn_depth=5,
               lstm_size=32,
               lstm_num_layers=2,
               lstm_keep_prob=1.0,
               tanh_constant=None,
               temperature=None,
               num_funcs=2,
               lr_init=1e-3,
               lr_dec_start=0,
               lr_dec_every=100,
               lr_dec_rate=0.9,
               l2_reg=0,
               entropy_weight=None,
               clip_mode=None,
               grad_bound=None,
               bl_dec=0.999,
               optim_algo="adam",
               sync_replicas=False,
               num_aggregate=None,
               num_replicas=None,
               name="controller"):

    print("-" * 80)
    print("Building PTBEnasController")

    self.rhn_depth = rhn_depth
    self.lstm_size = lstm_size
    self.lstm_num_layers = lstm_num_layers 
    self.lstm_keep_prob = lstm_keep_prob
    self.tanh_constant = tanh_constant
    self.temperature = temperature
    self.num_funcs = num_funcs
    self.lr_init = lr_init
    self.lr_dec_start = lr_dec_start
    self.lr_dec_every = lr_dec_every
    self.lr_dec_rate = lr_dec_rate
    self.l2_reg = l2_reg
    self.entropy_weight = entropy_weight
    self.clip_mode = clip_mode
    self.grad_bound = grad_bound
    self.bl_dec = bl_dec
    self.optim_algo = optim_algo
    self.sync_replicas = sync_replicas
    self.num_aggregate = num_aggregate
    self.num_replicas = num_replicas
    self.name = name

    self._create_params()
    self._build_sampler()

  def _create_params(self):
    initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
    with tf.variable_scope(self.name, initializer=initializer):
      with tf.variable_scope("lstm"):
        self.w_lstm = []
        for layer_id in range(self.lstm_num_layers):
          with tf.variable_scope("layer_{}".format(layer_id)):
            w = tf.get_variable("w", [2 * self.lstm_size, 4 * self.lstm_size])
            self.w_lstm.append(w)

      num_funcs = self.num_funcs
      with tf.variable_scope("embedding"):
        self.g_emb = tf.get_variable("g_emb", [1, self.lstm_size])
        self.w_emb = tf.get_variable("w", [num_funcs, self.lstm_size])

      with tf.variable_scope("softmax"):
        self.w_soft = tf.get_variable("w", [self.lstm_size, num_funcs])

      with tf.variable_scope("attention"):
        self.attn_w_1 = tf.get_variable("w_1", [self.lstm_size, self.lstm_size])
        self.attn_w_2 = tf.get_variable("w_2", [self.lstm_size, self.lstm_size])
        self.attn_v = tf.get_variable("v", [self.lstm_size, 1])

  def _build_sampler(self):
    """Build the sampler ops and the log_prob ops."""

    arc_seq = []
    sample_log_probs = []
    sample_entropy = []
    all_h = []
    all_h_w = []

    # sampler ops
    inputs = self.g_emb
    prev_c, prev_h = [], []
    for _ in range(self.lstm_num_layers):
      prev_c.append(tf.zeros([1, self.lstm_size], dtype=tf.float32))
      prev_h.append(tf.zeros([1, self.lstm_size], dtype=tf.float32))

    # used = tf.zeros([self.rhn_depth, 2], dtype=tf.int32)
    for layer_id in range(self.rhn_depth):
      next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
      prev_c, prev_h = next_c, next_h
      all_h.append(next_h[-1])
      all_h_w.append(tf.matmul(next_h[-1], self.attn_w_1))

      if layer_id > 0:
        query = tf.matmul(next_h[-1], self.attn_w_2)
        query = query + tf.concat(all_h_w[:-1], axis=0)
        query = tf.tanh(query)
        logits = tf.matmul(query, self.attn_v)
        logits = tf.reshape(logits, [1, layer_id])

        if self.temperature is not None:
          logits /= self.temperature
        if self.tanh_constant is not None:
          logits = self.tanh_constant * tf.tanh(logits)
        diff = tf.to_float(layer_id - tf.range(0, layer_id)) ** 2
        logits -= tf.reshape(diff, [1, layer_id]) / 6.0

        skip_index = tf.multinomial(logits, 1)
        skip_index = tf.to_int32(skip_index)
        skip_index = tf.reshape(skip_index, [1])
        arc_seq.append(skip_index)

        log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=skip_index)
        sample_log_probs.append(log_prob)

        entropy = log_prob * tf.exp(-log_prob)
        sample_entropy.append(tf.stop_gradient(entropy))

        inputs = tf.nn.embedding_lookup(
          tf.concat(all_h[:-1], axis=0), skip_index)
        inputs /= (0.1 + tf.to_float(layer_id - skip_index))
      else:
        inputs = self.g_emb

      next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
      prev_c, prev_h = next_c, next_h
      logits = tf.matmul(next_h[-1], self.w_soft)
      if self.temperature is not None:
        logits /= self.temperature
      if self.tanh_constant is not None:
        logits = self.tanh_constant * tf.tanh(logits)
      func = tf.multinomial(logits, 1)
      func = tf.to_int32(func)
      func = tf.reshape(func, [1])
      arc_seq.append(func)
      log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=func)
      sample_log_probs.append(log_prob)
      entropy = log_prob * tf.exp(-log_prob)
      sample_entropy.append(tf.stop_gradient(entropy))
      inputs = tf.nn.embedding_lookup(self.w_emb, func)

    arc_seq = tf.concat(arc_seq, axis=0)
    self.sample_arc = arc_seq

    self.sample_log_probs = tf.concat(sample_log_probs, axis=0)
    self.ppl = tf.exp(tf.reduce_mean(self.sample_log_probs))

    sample_entropy = tf.concat(sample_entropy, axis=0)
    self.sample_entropy = tf.reduce_sum(sample_entropy)

    self.all_h = all_h

  def build_trainer(self):
    # actor
    self.valid_loss = tf.placeholder(dtype=tf.float32, shape=[])

    #self.valid_loss = tf.to_float(child_model.rl_loss)
    #self.valid_loss = tf.stop_gradient(self.valid_loss)
    self.valid_ppl = tf.exp(self.valid_loss)
    self.reward = 80.0 / self.valid_ppl

    if self.entropy_weight is not None:
      self.reward += self.entropy_weight * self.sample_entropy

    # or baseline
    self.sample_log_probs = tf.reduce_sum(self.sample_log_probs)
    self.baseline = tf.Variable(0.0, dtype=tf.float32, trainable=False)
    baseline_update = tf.assign_sub(
      self.baseline, (1 - self.bl_dec) * (self.baseline - self.reward))

    with tf.control_dependencies([baseline_update]):
      self.reward = tf.identity(self.reward)
    self.loss = self.sample_log_probs * (self.reward - self.baseline)

    self.train_step = tf.Variable(
        0, dtype=tf.int32, trainable=False, name="train_step")
    tf_variables = [var
        for var in tf.trainable_variables() if var.name.startswith(self.name)]

    self.train_op, self.lr, self.grad_norm, self.optimizer = get_train_ops(
      self.loss,
      tf_variables,
      self.train_step,
      clip_mode=self.clip_mode,
      grad_bound=self.grad_bound,
      l2_reg=self.l2_reg,
      lr_init=self.lr_init,
      lr_dec_start=self.lr_dec_start,
      lr_dec_every=self.lr_dec_every,
      lr_dec_rate=self.lr_dec_rate,
      optim_algo=self.optim_algo,
      sync_replicas=self.sync_replicas,
      num_aggregate=self.num_aggregate,
      num_replicas=self.num_replicas)

