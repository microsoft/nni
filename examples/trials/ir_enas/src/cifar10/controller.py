import sys
import os
import time

import numpy as np
import tensorflow as tf

from src.controller import Controller
from src.utils import get_train_ops
from src.common_ops import stack_lstm

from tensorflow.python.training import moving_averages

class ConvController(Controller):
  def __init__(self,
               num_branches=6,
               num_layers=4,
               num_blocks_per_branch=8,
               lstm_size=32,
               lstm_num_layers=2,
               lstm_keep_prob=1.0,
               tanh_constant=None,
               temperature=None,
               lr_init=1e-3,
               lr_dec_start=0,
               lr_dec_every=100,
               lr_dec_rate=0.9,
               l2_reg=0,
               clip_mode=None,
               grad_bound=None,
               use_critic=False,
               bl_dec=0.999,
               optim_algo="adam",
               sync_replicas=False,
               num_aggregate=None,
               num_replicas=None,
               name="controller"):

    print "-" * 80
    print "Building ConvController"

    self.num_branches = num_branches
    self.num_layers = num_layers
    self.num_blocks_per_branch = num_blocks_per_branch
    self.lstm_size = lstm_size
    self.lstm_num_layers = lstm_num_layers 
    self.lstm_keep_prob = lstm_keep_prob
    self.tanh_constant = tanh_constant
    self.temperature = temperature
    self.lr_init = lr_init
    self.lr_dec_start = lr_dec_start
    self.lr_dec_every = lr_dec_every
    self.lr_dec_rate = lr_dec_rate
    self.l2_reg = l2_reg
    self.clip_mode = clip_mode
    self.grad_bound = grad_bound
    self.use_critic = use_critic
    self.bl_dec = bl_dec
    self.optim_algo = optim_algo
    self.sync_replicas = sync_replicas
    self.num_aggregate = num_aggregate
    self.num_replicas = num_replicas
    self.name = name

    self._create_params()
    self._build_sampler()

  def _create_params(self):
    with tf.variable_scope(self.name):
      with tf.variable_scope("lstm"):
        self.w_lstm = []
        for layer_id in xrange(self.lstm_num_layers):
          with tf.variable_scope("layer_{}".format(layer_id)):
            w = tf.get_variable("w", [2 * self.lstm_size, 4 * self.lstm_size])
            self.w_lstm.append(w)

      self.num_configs = (2 ** self.num_blocks_per_branch) - 1
      with tf.variable_scope("embedding"):
        self.g_emb = tf.get_variable("g_emb", [1, self.lstm_size])
        self.w_emb = tf.get_variable("w", [self.num_blocks_per_branch,
                                           self.lstm_size])

      with tf.variable_scope("softmax"):
        self.w_soft = tf.get_variable("w", [self.lstm_size,
                                            self.num_blocks_per_branch])
      with tf.variable_scope("critic"):
        self.w_critic = tf.get_variable("w", [self.lstm_size, 1])

  def _build_sampler(self):
    """Build the sampler ops and the log_prob ops."""

    arc_seq = []
    sample_log_probs = []
    all_h = []

    # sampler ops
    inputs = self.g_emb
    prev_c = [tf.zeros([1, self.lstm_size], dtype=tf.float32)
              for _ in xrange(self.lstm_num_layers)]
    prev_h = [tf.zeros([1, self.lstm_size], dtype=tf.float32)
              for _ in xrange(self.lstm_num_layers)]
    for layer_id in xrange(self.num_layers):
      for branch_id in xrange(self.num_branches):
        next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
        all_h.append(tf.stop_gradient(next_h[-1]))

        logits = tf.matmul(next_h[-1], self.w_soft)
        if self.temperature is not None:
          logits /= self.temperature
        if self.tanh_constant is not None:
          logits = self.tanh_constant * tf.tanh(logits)

        config_id = tf.multinomial(logits, 1)
        config_id = tf.to_int32(config_id)
        config_id = tf.reshape(config_id, [1])
        arc_seq.append(config_id)
        log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=config_id)
        sample_log_probs.append(log_prob)

        inputs = tf.nn.embedding_lookup(self.w_emb, config_id)
    arc_seq = tf.concat(arc_seq, axis=0)
    self.sample_arc = arc_seq

    self.sample_log_probs = tf.concat(sample_log_probs, axis=0)
    self.ppl = tf.exp(tf.reduce_sum(self.sample_log_probs) /
                      tf.to_float(self.num_layers * self.num_branches))
    self.all_h = all_h

  def build_trainer(self, child_model):
    # actor
    child_model.build_valid_rl()
    self.valid_acc = (tf.to_float(child_model.valid_shuffle_acc) /
                      tf.to_float(child_model.batch_size))
    self.reward = self.valid_acc

    if self.use_critic:
      # critic
      all_h = tf.concat(self.all_h, axis=0)
      value_function = tf.matmul(all_h, self.w_critic)
      advantage = value_function - self.reward
      critic_loss = tf.reduce_sum(advantage ** 2)
      self.baseline = tf.reduce_mean(value_function)
      self.loss = -tf.reduce_mean(self.sample_log_probs * advantage)

      critic_train_step = tf.Variable(
          0, dtype=tf.int32, trainable=False, name="critic_train_step")
      critic_train_op, _, _, _ = get_train_ops(
        critic_loss,
        [self.w_critic],
        critic_train_step,
        clip_mode=None,
        lr_init=1e-3,
        lr_dec_start=0,
        lr_dec_every=int(1e9),
        optim_algo="adam",
        sync_replicas=False)
    else:
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
    tf_variables = [var for var in tf.trainable_variables()
                    if var.name.startswith(self.name)
                      and "w_critic" not in var.name]
    print "-" * 80
    for var in tf_variables:
      print var
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

    if self.use_critic:
      self.train_op = tf.group(self.train_op, critic_train_op)

