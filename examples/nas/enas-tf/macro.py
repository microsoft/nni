# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, GlobalAveragePooling2D

from nni.nas.tensorflow import mutables
from ops import FactorizedReduce, ConvBranch, PoolBranch


class ENASLayer(mutables.MutableScope):

    def __init__(self, key, prev_labels, in_filters, out_filters):
        super().__init__(key)
        self.mutable = mutables.LayerChoice([
            ConvBranch(in_filters, out_filters, 3, 1, 1, separable=False),
            ConvBranch(in_filters, out_filters, 3, 1, 1, separable=True),
            ConvBranch(in_filters, out_filters, 5, 1, 2, separable=False),
            ConvBranch(in_filters, out_filters, 5, 1, 2, separable=True),
            PoolBranch('avg', in_filters, out_filters, 3, 1, 1),
            PoolBranch('max', in_filters, out_filters, 3, 1, 1)
        ])
        if len(prev_labels) > 0:
            self.skipconnect = mutables.InputChoice(choose_from=prev_labels, n_chosen=None)
        else:
            self.skipconnect = None
        self.batch_norm = BatchNormalization(trainable=False)

    def call(self, prev_layers):
        out = self.mutable(prev_layers[-1])
        if self.skipconnect is not None:
            connection = self.skipconnect(prev_layers[:-1])
            if connection is not None:
                out += connection
        return self.batch_norm(out)


class GeneralNetwork(Model):
    def __init__(self, num_layers=12, out_filters=24, in_channels=3, num_classes=10,
                 dropout_rate=0.0):
        super().__init__()
        self.num_layers = num_layers

        self.stem = Sequential([
            Conv2D(out_filters, kernel_size=3, strides=1, use_bias=False), # FIXME: padding=1
            BatchNormalization()
        ])

        pool_distance = self.num_layers // 3
        self.pool_layers_idx = [pool_distance - 1, 2 * pool_distance - 1]
        self.dropout_rate = dropout_rate
        self.dropout = Dropout(self.dropout_rate)

        self.enas_layers = []
        self.pool_layers = []
        labels = []
        for layer_id in range(self.num_layers):
            labels.append("layer_{}".format(layer_id))
            if layer_id in self.pool_layers_idx:
                self.pool_layers.append(FactorizedReduce(out_filters, out_filters))
            self.enas_layers.append(ENASLayer(labels[-1], labels[:-1], out_filters, out_filters))

        self.gap = GlobalAveragePooling2D()
        self.dense = Dense(num_classes)

    def call(self, x):
        bs = x.shape[0]
        cur = self.stem(x)

        layers = [cur]

        for layer_id in range(self.num_layers):
            cur = self.enas_layers[layer_id](layers)
            layers.append(cur)
            if layer_id in self.pool_layers_idx:
                for i, layer in enumerate(layers):
                    layers[i] = self.pool_layers[self.pool_layers_idx.index(layer_id)](layer)
                cur = layers[-1]

        cur = tf.reshape(self.gap(cur), [bs, -1])
        cur = self.dropout(cur)
        logits = self.dense(cur)
        return logits
