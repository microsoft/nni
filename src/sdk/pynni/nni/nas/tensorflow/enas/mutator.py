# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Embedding, LSTMCell, RNN
from tensorflow.keras.losses import SparseCategoricalCrossentropy, Reduction

from nni.nas.tensorflow.mutator import Mutator
from nni.nas.tensorflow.mutables import LayerChoice, InputChoice, MutableScope


class StackedLSTMCell(Model):
    def __init__(self, layers, size):
        super().__init__()
        self.lstm_layer_num = layers
        self._size = size
        self.lstm_modules = [LSTMCell(units=size, use_bias=False) for _ in range(layers)]

    def reset_states(self):
        self._states = [(tf.zeros((1, self._size)), tf.zeros((1, self._size))) for _ in range(self.lstm_layer_num)]

    def call(self, x):
        for i, m in enumerate(self.lstm_modules):
            x, self._states[i] = m(x, self._states[i])
        return x


class EnasMutator(Mutator):
    def __init__(
            self,
            model,
            lstm_size=64,
            lstm_num_layers=1,
            tanh_constant=1.5,
            cell_exit_extra_step=False,
            skip_target=0.4,
            temperature=None,
            branch_bias=0.25,
            entropy_reduction='sum'):
        super().__init__(model)
        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        self.tanh_constant = tanh_constant
        self.temperature = temperature
        self.cell_exit_extra_step = cell_exit_extra_step
        self.skip_target = skip_target
        self.branch_bias = branch_bias
        self.lstm = StackedLSTMCell(lstm_num_layers, lstm_size)
        self.attn_anchor = Dense(self.lstm_size, use_bias=False)
        self.attn_query = Dense(self.lstm_size, use_bias=False)
        self.v_attn = Dense(1, use_bias=False)
        self.g_emb = tf.Variable(tf.random.normal((1, self.lstm_size)) * 0.1)
        self.skip_targets = tf.constant([1.0 - self.skip_target, self.skip_target])
        assert entropy_reduction in ['sum', 'mean'], 'Entropy reduction must be one of sum and mean.'
        self.entropy_reduction = tf.reduce_sum if entropy_reduction == 'sum' else tf.reduce_mean
        self.cross_entropy_loss = SparseCategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE)
        self.bias_dict = {}

        self.max_layer_choice = 0
        for mutable in self.mutables:
            if isinstance(mutable, LayerChoice):
                if self.max_layer_choice == 0:
                    self.max_layer_choice = mutable.length
                assert self.max_layer_choice == mutable.length, \
                        "ENAS mutator requires all layer choice have the same number of candidates."
                if 'reduce' in mutable.key:
                    bias = []
                    for choice in mutable.choices:
                        if 'conv' in str(type(choice)).lower():
                            bias.append(self.branch_bias)
                        else:
                            bias.append(-self.branch_bias)
                    self.bias_dict[mutable.key] = tf.constant(bias)

        self.embedding = Embedding(self.max_layer_choice + 1, self.lstm_size)
        self.soft = Dense(self.max_layer_choice, use_bias=False)

    def sample_search(self):
        self._initialize()
        self._sample(self.mutables)
        return self._choices

    def sample_final(self):
        return self.sample_search()

    def _sample(self, tree):
        mutable = tree.mutable
        if isinstance(mutable, LayerChoice) and mutable.key not in self._choices:
            self._choices[mutable.key] = self._sample_layer_choice(mutable)
        elif isinstance(mutable, InputChoice) and mutable.key not in self._choices:
            self._choices[mutable.key] = self._sample_input_choice(mutable)
        for child in tree.children:
            self._sample(child)
        if self.cell_exit_extra_step and isinstance(mutable, MutableScope) and mutable.key not in self._anchors_hid:
            self._anchors_hid[mutable.key] = self.lstm(self._inputs)

    def _initialize(self):
        self._choices = {}
        self._anchors_hid = {}
        self._inputs = tf.Variable(self.g_emb)
        #self.lstm.build(input_shape=(1, lstm_size))
        self.lstm.reset_states()
        self._h = [tf.zeros((1, self.lstm_size), dtype=self._inputs.dtype) for _ in range(self.lstm_num_layers)]
        self.sample_log_prob = 0
        self.sample_entropy = 0
        self.sample_skip_penalty = 0

    def _sample_layer_choice(self, mutable):
        logit = self.soft(self.lstm(self._inputs))
        if self.temperature is not None:
            logit /= self.temperature
        if self.tanh_constant is not None:
            logit = self.tanh_constant * tf.tanh(logit)
        if mutable.key in self.bias_dict:
            logit += self.bias_dict[mutable.key]
        branch_id = tf.reshape(tf.random.categorical(tf.nn.softmax(logit, axis=-1), 1), [-1])
        log_prob = self.cross_entropy_loss(branch_id, logit)
        self.sample_log_prob += self.entropy_reduction(log_prob)
        entropy = log_prob * tf.math.exp(-log_prob)
        self.sample_entropy += self.entropy_reduction(entropy)
        self._inputs = self.embedding(branch_id)
        return tf.reshape(tf.cast(tf.one_hot(branch_id, self.max_layer_choice), tf.bool), -1)

    def _sample_input_choice(self, mutable):
        query, anchors = [], []
        for label in mutable.choose_from:
            if label not in self._anchors_hid:
                self._anchors_hid[label] = self.lstm(self._inputs)
            query.append(self.attn_anchor(self._anchors_hid[label]))
            anchors.append(self._anchors_hid[label])
        query = tf.concat(query, 0)
        query = tf.tanh(query + self.attn_query(anchors[-1]))
        query = self.v_attn(query)
        if self.temperature is not None:
            query /= self.temperature
        if self.tanh_constant is not None:
            query = self.tanh_constant * tf.tanh(query)

        if mutable.n_chosen is None:
            logit = tf.concat([-query, query], 1)

            skip = tf.reshape(tf.random.categorical(tf.nn.softmax(logit, axis=-1), 1), [-1])
            skip_prob = tf.math.sigmoid(logit)
            kl = tf.reduce_sum(skip_prob * tf.math.log(skip_prob / self.skip_targets))
            self.sample_skip_penalty += kl
            log_prob = self.cross_entropy_loss(skip, logit)

            skip = tf.cast(skip, tf.float32)
            self._inputs = tf.expand_dims(tf.tensordot(skip, tf.concat(anchors, 0), 1) / (1. + tf.reduce_sum(skip)), 0)
        else:
            assert mutable.n_chosen == 1, "Input choice must select exactly one or any in ENAS."
            logit = tf.reshape(query, [1, -1])
            index = tf.reshape(tf.random.categorical(tf.nn.softmax(logit, axis=-1), 1), [-1])
            skip = tf.reshape(tf.one_hot(index, mutable.n_candidates), [-1])
            log_prob = self.cross_entropy_loss(index, logit)
            self._inputs = anchors[index.item()]

        self.sample_log_prob += self.entropy_reduction(log_prob)
        entropy = log_prob * tf.exp(-log_prob)
        self.sample_entropy += self.entropy_reduction(entropy)
        return tf.cast(skip, tf.bool)
