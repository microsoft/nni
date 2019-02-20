from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import tensorflow as tf

from src.common_ops import lstm

from src.utils import count_model_params
from src.utils import get_train_ops

from src.ptb.data_utils import ptb_input_producer
from src.ptb.ptb_ops import batch_norm
from src.ptb.ptb_ops import layer_norm


class PTBEnasChild(object):
  def __init__(self,
               x_train,
               x_valid,
               x_test,
               num_funcs=4,
               rnn_l2_reg=None,
               rnn_slowness_reg=None,
               rhn_depth=2,
               fixed_arc=None,
               base_number=4,
               batch_size=32,
               bptt_steps=25,
               lstm_num_layers=2,
               lstm_hidden_size=32,
               lstm_e_keep=1.0,
               lstm_x_keep=1.0,
               lstm_h_keep=1.0,
               lstm_o_keep=1.0,
               lstm_l_skip=False,
               vocab_size=10000,
               lr_warmup_val=None,
               lr_warmup_steps=None,
               lr_init=1.0,
               lr_dec_start=4,
               lr_dec_every=1,
               lr_dec_rate=0.5,
               lr_dec_min=None,
               l2_reg=None,
               clip_mode="global",
               grad_bound=5.0,
               optim_algo=None,
               optim_moving_average=None,
               sync_replicas=False,
               num_aggregate=None,
               num_replicas=None,
               temperature=None,
               name="ptb_lstm",
               seed=None,
               *args,
               **kwargs):
    """
    Args:
      lr_dec_every: number of epochs to decay
    """
    print("-" * 80)
    print("Build model {}".format(name))

    self.num_funcs = num_funcs
    self.rnn_l2_reg = rnn_l2_reg
    self.rnn_slowness_reg = rnn_slowness_reg
    self.rhn_depth = rhn_depth
    self.fixed_arc = fixed_arc
    self.base_number = base_number
    self.num_nodes = 2 * self.base_number - 1
    self.batch_size = batch_size
    self.bptt_steps = bptt_steps
    self.lstm_num_layers = lstm_num_layers
    self.lstm_hidden_size = lstm_hidden_size
    self.lstm_e_keep = lstm_e_keep
    self.lstm_x_keep = lstm_x_keep
    self.lstm_h_keep = lstm_h_keep
    self.lstm_o_keep = lstm_o_keep
    self.lstm_l_skip = lstm_l_skip
    self.vocab_size = vocab_size
    self.lr_warmup_val = lr_warmup_val
    self.lr_warmup_steps = lr_warmup_steps
    self.lr_init = lr_init
    self.lr_dec_min = lr_dec_min
    self.l2_reg = l2_reg
    self.clip_mode = clip_mode
    self.grad_bound = grad_bound

    self.optim_algo = optim_algo
    self.optim_moving_average = optim_moving_average
    self.sync_replicas = sync_replicas
    self.num_aggregate = num_aggregate
    self.num_replicas = num_replicas
    self.temperature = temperature

    self.name = name
    self.seed = seed
    
    self.global_step = None
    self.valid_loss = None
    self.test_loss = None

    print("Build data ops")
    # training data
    self.x_train, self.y_train, self.num_train_batches = ptb_input_producer(
      x_train, self.batch_size, self.bptt_steps)
    self.y_train = tf.reshape(self.y_train, [self.batch_size * self.bptt_steps])

    self.lr_dec_start = lr_dec_start * self.num_train_batches
    self.lr_dec_every = lr_dec_every * self.num_train_batches
    self.lr_dec_rate = lr_dec_rate

    # valid data
    self.x_valid, self.y_valid, self.num_valid_batches = ptb_input_producer(
      np.copy(x_valid), self.batch_size, self.bptt_steps)
    self.y_valid = tf.reshape(self.y_valid, [self.batch_size * self.bptt_steps])

    # valid_rl data
    (self.x_valid_rl, self.y_valid_rl,
     self.num_valid_batches) = ptb_input_producer(
       np.copy(x_valid), self.batch_size, self.bptt_steps, randomize=True)
    self.y_valid_rl = tf.reshape(self.y_valid_rl,
                                 [self.batch_size * self.bptt_steps])

    # test data
    self.x_test, self.y_test, self.num_test_batches = ptb_input_producer(
      x_test, 1, 1)
    self.y_test = tf.reshape(self.y_test, [1])

    self.x_valid_raw = x_valid

  def eval_once(self, sess, eval_set, feed_dict=None, verbose=False):
    """Expects self.acc and self.global_step to be defined.

    Args:
      sess: tf.Session() or one of its wrap arounds.
      feed_dict: can be used to give more information to sess.run().
      eval_set: "valid" or "test"
    """

    assert self.global_step is not None, "TF op self.global_step not defined."
    global_step = sess.run(self.global_step)
    print("Eval at {}".format(global_step))
   
    if eval_set == "valid":
      assert self.valid_loss is not None, "TF op self.valid_loss is not defined."
      num_batches = self.num_valid_batches
      loss_op = self.valid_loss
      reset_op = self.valid_reset
      batch_size = self.batch_size
      bptt_steps = self.bptt_steps
    elif eval_set == "test":
      assert self.test_loss is not None, "TF op self.test_loss is not defined."
      num_batches = self.num_test_batches
      loss_op = self.test_loss
      reset_op = self.test_reset
      batch_size = 1
      bptt_steps = 1
    else:
      raise ValueError("Unknown eval_set '{}'".format(eval_set))

    sess.run(reset_op)
    total_loss = 0
    for batch_id in range(num_batches):
      curr_loss = sess.run(loss_op, feed_dict=feed_dict)
      total_loss += curr_loss  # np.minimum(curr_loss, 10.0 * bptt_steps * batch_size)
      ppl_sofar = np.exp(total_loss / (bptt_steps * batch_size * (batch_id + 1)))
      if verbose and (batch_id + 1) % 1000 == 0:
        print("{:<5d} {:<6.2f}".format(batch_id + 1, ppl_sofar))
    if verbose:
      print("")
    log_ppl = total_loss / (num_batches * batch_size * bptt_steps)
    ppl = np.exp(np.minimum(log_ppl, 10.0))
    sess.run(reset_op)
    print("{}_total_loss: {:<6.2f}".format(eval_set, total_loss))
    print("{}_log_ppl: {:<6.2f}".format(eval_set, log_ppl))
    print("{}_ppl: {:<6.2f}".format(eval_set, ppl))
    return ppl

  def _build_train(self):
    print("Build train graph")
    all_h, self.train_reset = self._model(self.x_train, True, False)
    log_probs = self._get_log_probs(
      all_h, self.y_train, batch_size=self.batch_size, is_training=True)
    self.loss = tf.reduce_sum(log_probs) / tf.to_float(self.batch_size)
    self.train_ppl = tf.exp(tf.reduce_mean(log_probs))

    tf_variables = [
      var for var in tf.trainable_variables() if var.name.startswith(self.name)]
    self.num_vars = count_model_params(tf_variables)
    print("-" * 80)
    print("Model has {} parameters".format(self.num_vars))

    loss = self.loss
    if self.rnn_l2_reg is not None:
      loss += (self.rnn_l2_reg * tf.reduce_sum(all_h ** 2) /
               tf.to_float(self.batch_size))
    if self.rnn_slowness_reg is not None:
      loss += (self.rnn_slowness_reg * self.all_h_diff /
               tf.to_float(self.batch_size))
    self.global_step = tf.Variable(
      0, dtype=tf.int32, trainable=False, name="global_step")
    (self.train_op,
     self.lr,
     self.grad_norm,
     self.optimizer,
     self.grad_norms) = get_train_ops(
       loss,
       tf_variables,
       self.global_step,
       clip_mode=self.clip_mode,
       grad_bound=self.grad_bound,
       l2_reg=self.l2_reg,
       lr_warmup_val=self.lr_warmup_val,
       lr_warmup_steps=self.lr_warmup_steps,
       lr_init=self.lr_init,
       lr_dec_start=self.lr_dec_start,
       lr_dec_every=self.lr_dec_every,
       lr_dec_rate=self.lr_dec_rate,
       lr_dec_min=self.lr_dec_min,
       optim_algo=self.optim_algo,
       moving_average=self.optim_moving_average,
       sync_replicas=self.sync_replicas,
       num_aggregate=self.num_aggregate,
       num_replicas=self.num_replicas,
       get_grad_norms=True,
     )

  def _get_log_probs(self, all_h, labels, batch_size=None, is_training=False):
    logits = tf.matmul(all_h, self.w_emb, transpose_b=True)
    log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels)

    return log_probs

  def _build_valid(self):
    print("-" * 80)
    print("Build valid graph")
    all_h, self.valid_reset = self._model(self.x_valid, False, False)
    all_h = tf.stop_gradient(all_h)
    log_probs = self._get_log_probs(all_h, self.y_valid)
    self.valid_loss = tf.reduce_sum(log_probs)

  def _build_valid_rl(self):
    print("-" * 80)
    print("Build valid graph for RL")
    all_h, self.valid_rl_reset = self._model(
      self.x_valid_rl, False, False, should_carry=False)
    all_h = tf.stop_gradient(all_h)
    log_probs = self._get_log_probs(all_h, self.y_valid_rl)
    self.rl_loss = tf.reduce_mean(log_probs)
    self.rl_loss = tf.stop_gradient(self.rl_loss)

  def _build_test(self):
    print("-" * 80)
    print("Build test graph")
    all_h, self.test_reset = self._model(self.x_test, False, True)
    all_h = tf.stop_gradient(all_h)
    log_probs = self._get_log_probs(all_h, self.y_test)
    self.test_loss = tf.reduce_sum(log_probs)

  def _rhn_fixed(self, x, prev_s, w_prev, w_skip, is_training,
                 x_mask=None, s_mask=None):
    batch_size = prev_s.get_shape()[0].value
    start_idx = self.sample_arc[0] * 2 * self.lstm_hidden_size
    end_idx = start_idx + 2 * self.lstm_hidden_size
    if is_training:
      assert x_mask is not None, "x_mask is None"
      assert s_mask is not None, "s_mask is None"
      ht = tf.matmul(tf.concat([x * x_mask, prev_s * s_mask], axis=1), w_prev)
    else:
      ht = tf.matmul(tf.concat([x, prev_s], axis=1), w_prev)
    # with tf.variable_scope("rhn_layer_0"):
    #   ht = layer_norm(ht, is_training)
    h, t = tf.split(ht, 2, axis=1)

    if self.sample_arc[0] == 0:
      h = tf.tanh(h)
    elif self.sample_arc[0] == 1:
      h = tf.nn.relu(h)
    elif self.sample_arc[0] == 2:
      h = tf.identity(h)
    elif self.sample_arc[0] == 3:
      h = tf.sigmoid(h)
    else:
      raise ValueError("Unknown func_idx {}".format(self.sample_arc[0]))
    t = tf.sigmoid(t)
    s = prev_s + t * (h - prev_s)
    layers = [s]

    start_idx = 1
    used = np.zeros([self.rhn_depth], dtype=np.int32)
    for rhn_layer_id in range(1, self.rhn_depth):
      with tf.variable_scope("rhn_layer_{}".format(rhn_layer_id)):
        prev_idx = self.sample_arc[start_idx]
        func_idx = self.sample_arc[start_idx + 1]
        used[prev_idx] = 1
        prev_s = layers[prev_idx]
        if is_training:
          ht = tf.matmul(prev_s * s_mask, w_skip[rhn_layer_id])
        else:
          ht = tf.matmul(prev_s, w_skip[rhn_layer_id])
        # ht = layer_norm(ht, is_training)
        h, t = tf.split(ht, 2, axis=1)

        if func_idx == 0:
          h = tf.tanh(h)
        elif func_idx == 1:
          h = tf.nn.relu(h)
        elif func_idx == 2:
          h = tf.identity(h)
        elif func_idx == 3:
          h = tf.sigmoid(h)
        else:
          raise ValueError("Unknown func_idx {}".format(func_idx))

        t = tf.sigmoid(t)
        s = prev_s + t * (h - prev_s)
        layers.append(s)
        start_idx += 2

    layers = [prev_layer for u, prev_layer in zip(used, layers) if u == 0]
    layers = tf.add_n(layers) / np.sum(1.0 - used)
    layers.set_shape([batch_size, self.lstm_hidden_size])

    return layers

  def _rhn_enas(self, x, prev_s, w_prev, w_skip, is_training,
                x_mask=None, s_mask=None):
    batch_size = prev_s.get_shape()[0].value
    start_idx = self.sample_arc[0] * 2 * self.lstm_hidden_size
    end_idx = start_idx + 2 * self.lstm_hidden_size
    # h = func(x*W1)
    # t = sigmoid(h*W2)
    if is_training:
      assert x_mask is not None, "x_mask is None"
      assert s_mask is not None, "s_mask is None"
      ht = tf.matmul(tf.concat([x * x_mask, prev_s * s_mask], axis=1),
                     w_prev[start_idx:end_idx, :])
    else:
      ht = tf.matmul(tf.concat([x, prev_s], axis=1),
                     w_prev[start_idx:end_idx, :])
    with tf.variable_scope("rhn_layer_0"):
      ht = batch_norm(ht, is_training)
    h, t = tf.split(ht, 2, axis=1)
    func_idx = self.sample_arc[0]
    h = tf.case(
      {
        tf.equal(func_idx, 0): lambda: tf.tanh(h),
        tf.equal(func_idx, 1): lambda: tf.nn.relu(h),
        tf.equal(func_idx, 2): lambda: tf.identity(h),
        tf.equal(func_idx, 3): lambda: tf.sigmoid(h),
      },
      default=lambda: tf.constant(0.0, dtype=tf.float32), exclusive=True)
    t = tf.sigmoid(t)
    s = prev_s + t * (h - prev_s)
    layers = [s]

    start_idx = 1
    used = []
    for rhn_layer_id in range(1, self.rhn_depth):
      with tf.variable_scope("rhn_layer_{}".format(rhn_layer_id)):
        prev_idx = self.sample_arc[start_idx]
        func_idx = self.sample_arc[start_idx + 1]
        # tf.one_hot(1, 4) == [0, 1, 0, 0]
        curr_used = tf.one_hot(prev_idx, depth=self.rhn_depth, dtype=tf.int32)
        used.append(curr_used)
        # since we share weight, we need to get the corresponding weights
        # every idx and func have a unique W (hash: )
        w_start = (prev_idx * self.num_funcs + func_idx) * self.lstm_hidden_size
        w_end = w_start + self.lstm_hidden_size
        w = w_skip[rhn_layer_id][w_start:w_end, :]
        prev_s = tf.concat(layers, axis=0)
        prev_s = prev_s[prev_idx*batch_size : (prev_idx+1)*batch_size, :]
        if is_training:
          ht = tf.matmul(prev_s * s_mask, w)
        else:
          ht = tf.matmul(prev_s, w)
        ht = batch_norm(ht, is_training)
        h, t = tf.split(ht, 2, axis=1)
        h = tf.case(
          {
            tf.equal(func_idx, 0): lambda: tf.tanh(h),
            tf.equal(func_idx, 1): lambda: tf.nn.relu(h),
            tf.equal(func_idx, 2): lambda: tf.identity(h),
            tf.equal(func_idx, 3): lambda: tf.sigmoid(h),
          },
          default=lambda: tf.constant(0.0, dtype=tf.float32), exclusive=True)
        t = tf.sigmoid(t)
        s = prev_s + t * (h - prev_s)
        layers.append(s)
        start_idx += 2

    used = tf.add_n(used)
    used = tf.equal(used, 0)
    with tf.control_dependencies([tf.Assert(tf.reduce_any(used), [used])]):
      layers = tf.stack(layers)
    layers = tf.boolean_mask(layers, used)
    layers = tf.reduce_mean(layers, axis=0)
    layers.set_shape([batch_size, self.lstm_hidden_size])
    layers = batch_norm(layers, is_training)
    
    return layers

  def _model(self, x, is_training, is_test, should_carry=True):
    if is_test:
      start_h = self.test_start_h
      num_steps = 1
      batch_size = 1
    else:
      start_h = self.start_h
      num_steps = self.bptt_steps
      batch_size = self.batch_size

    all_h = tf.TensorArray(tf.float32, size=num_steps, infer_shape=True)
    embedding = tf.nn.embedding_lookup(self.w_emb, x)

    if is_training:
      def _gen_mask(shape, keep_prob):
        _mask = tf.random_uniform(shape, dtype=tf.float32)
        _mask = tf.floor(_mask + keep_prob) / keep_prob
        return _mask

      # variational dropout in the embedding layer
      e_mask = _gen_mask([batch_size, num_steps], self.lstm_e_keep)
      first_e_mask = e_mask
      zeros = tf.zeros_like(e_mask)
      ones = tf.ones_like(e_mask)
      r = [tf.constant([[False]] * batch_size, dtype=tf.bool)]  # more zeros to e_mask
      for step in range(1, num_steps):
        should_zero = tf.logical_and(
          tf.equal(x[:, :step], x[:, step:step+1]),
          tf.equal(e_mask[:, :step], 0))
        should_zero = tf.reduce_any(should_zero, axis=1, keep_dims=True)
        r.append(should_zero)
      r = tf.concat(r, axis=1)
      e_mask = tf.where(r, tf.zeros_like(e_mask), e_mask)
      e_mask = tf.reshape(e_mask, [batch_size, num_steps, 1])
      embedding *= e_mask

      # variational dropout in the hidden layers
      x_mask, h_mask = [], []
      for layer_id in range(self.lstm_num_layers):
        x_mask.append(_gen_mask([batch_size, self.lstm_hidden_size], self.lstm_x_keep))
        h_mask.append(_gen_mask([batch_size, self.lstm_hidden_size], self.lstm_h_keep))
        h_mask.append(h_mask)

      # variational dropout in the output layer
      o_mask = _gen_mask([batch_size, self.lstm_hidden_size], self.lstm_o_keep)

    def condition(step, *args):
      return tf.less(step, num_steps)

    def body(step, prev_h, all_h):
      with tf.variable_scope(self.name):
        next_h = []
        for layer_id, (p_h, w_prev, w_skip) in enumerate(zip(prev_h, self.w_prev, self.w_skip)):
          with tf.variable_scope("layer_{}".format(layer_id)):
            if layer_id == 0:
              inputs = embedding[:, step, :]
            else:
              inputs = next_h[-1]

            if self.fixed_arc is None:
              curr_h = self._rhn_enas(
                inputs, p_h, w_prev, w_skip, is_training,
                x_mask=x_mask[layer_id] if is_training else None,
                s_mask=h_mask[layer_id] if is_training else None)
            else:
              curr_h = self._rhn_fixed(
                inputs, p_h, w_prev, w_skip, is_training,
                x_mask=x_mask[layer_id] if is_training else None,
                s_mask=h_mask[layer_id] if is_training else None)

            if self.lstm_l_skip:
              curr_h += inputs

            next_h.append(curr_h)

        out_h = next_h[-1]
        if is_training:
          out_h *= o_mask
        all_h = all_h.write(step, out_h)
      return step + 1, next_h, all_h
    
    loop_vars = [tf.constant(0, dtype=tf.int32), start_h, all_h]
    loop_outputs = tf.while_loop(condition, body, loop_vars, back_prop=True)
    next_h = loop_outputs[-2]
    all_h = loop_outputs[-1].stack()
    all_h_diff = (all_h[1:, :, :] - all_h[:-1, :, :]) ** 2
    self.all_h_diff = tf.reduce_sum(all_h_diff)
    all_h = tf.transpose(all_h, [1, 0, 2])
    all_h = tf.reshape(all_h, [batch_size * num_steps, self.lstm_hidden_size])
    
    carry_states = []
    reset_states = []
    for layer_id, (s_h, n_h) in enumerate(zip(start_h, next_h)):
      reset_states.append(tf.assign(s_h, tf.zeros_like(s_h), use_locking=True))
      carry_states.append(tf.assign(s_h, tf.stop_gradient(n_h), use_locking=True))

    if should_carry:
      with tf.control_dependencies(carry_states):
        all_h = tf.identity(all_h)

    return all_h, reset_states

  def _build_params(self):
    if self.lstm_hidden_size <= 300:
      init_range = 0.1
    elif self.lstm_hidden_size <= 400:
      init_range = 0.05
    else:
      init_range = 0.04
    initializer = tf.random_uniform_initializer(
      minval=-init_range, maxval=init_range)
    with tf.variable_scope(self.name, initializer=initializer):
      if self.fixed_arc is None:
        with tf.variable_scope("rnn"):
          self.w_prev, self.w_skip = [], []
          for layer_id in range(self.lstm_num_layers):
            with tf.variable_scope("layer_{}".format(layer_id)):
              w_prev = tf.get_variable(
                "w_prev",
                [2 * self.num_funcs * self.lstm_hidden_size,
                 2 * self.lstm_hidden_size])
              w_skip = [None]
              for rhn_layer_id in range(1, self.rhn_depth):
                with tf.variable_scope("layer_{}".format(rhn_layer_id)):
                  w = tf.get_variable(
                    "w", [self.num_funcs * rhn_layer_id * self.lstm_hidden_size,
                          2 * self.lstm_hidden_size])
                  w_skip.append(w)
              self.w_prev.append(w_prev)
              self.w_skip.append(w_skip)
      else:
        with tf.variable_scope("rnn"):
          self.w_prev, self.w_skip = [], []
          for layer_id in range(self.lstm_num_layers):
            with tf.variable_scope("layer_{}".format(layer_id)):
              w_prev = tf.get_variable("w_prev", [2 * self.lstm_hidden_size,
                                                  2 * self.lstm_hidden_size])
              w_skip = [None]
              for rhn_layer_id in range(1, self.rhn_depth):
                with tf.variable_scope("layer_{}".format(rhn_layer_id)):
                  w = tf.get_variable("w", [self.lstm_hidden_size,
                                            2 * self.lstm_hidden_size])
                  w_skip.append(w)
              self.w_prev.append(w_prev)
              self.w_skip.append(w_skip)

      with tf.variable_scope("embedding"):
        self.w_emb = tf.get_variable(
          "w", [self.vocab_size, self.lstm_hidden_size])

      with tf.variable_scope("starting_states"):
        zeros = np.zeros(
          [self.batch_size, self.lstm_hidden_size], dtype=np.float32)
        zeros_one_instance = np.zeros(
          [1, self.lstm_hidden_size], dtype=np.float32)

        self.start_h, self.test_start_h = [], []
        for _ in range(self.lstm_num_layers):
          self.start_h.append(tf.Variable(zeros, trainable=False))
          self.test_start_h.append(tf.Variable(zeros_one_instance,
                                               trainable=False))

  def connect_controller(self):

    '''
    if self.fixed_arc is None:
      self.sample_arc = controller_model.sample_arc
    else:
      sample_arc = np.array(
        [x for x in self.fixed_arc.split(" ") if x], dtype=np.int32)
      self.sample_arc = sample_arc
    '''
    self.sample_arc = tf.placeholder(tf.int32, shape=[None])

    self._build_params()
    self._build_train()
    self._build_valid()
    self._build_valid_rl()
    self._build_test()


