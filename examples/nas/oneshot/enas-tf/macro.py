# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    MaxPool2D,
    ReLU,
    SeparableConv2D,
)

from nni.nas.tensorflow.mutables import InputChoice, LayerChoice, MutableScope


def build_conv(filters, kernel_size, name=None):
    return Sequential([
        Conv2D(filters, kernel_size=1, use_bias=False),
        BatchNormalization(trainable=False),
        ReLU(),
        Conv2D(filters, kernel_size, padding='same'),
        BatchNormalization(trainable=False),
        ReLU(),
    ], name)

def build_separable_conv(filters, kernel_size, name=None):
    return Sequential([
        Conv2D(filters, kernel_size=1, use_bias=False),
        BatchNormalization(trainable=False),
        ReLU(),
        SeparableConv2D(filters, kernel_size, padding='same', use_bias=False),
        Conv2D(filters, kernel_size=1, use_bias=False),
        BatchNormalization(trainable=False),
        ReLU(),
    ], name)

def build_avg_pool(filters, name=None):
    return Sequential([
        Conv2D(filters, kernel_size=1, use_bias=False),
        BatchNormalization(trainable=False),
        ReLU(),
        AveragePooling2D(pool_size=3, strides=1, padding='same'),
        BatchNormalization(trainable=False),
    ], name)

def build_max_pool(filters, name=None):
    return Sequential([
        Conv2D(filters, kernel_size=1, use_bias=False),
        BatchNormalization(trainable=False),
        ReLU(),
        MaxPool2D(pool_size=3, strides=1, padding='same'),
        BatchNormalization(trainable=False),
    ], name)


class FactorizedReduce(Model):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = Conv2D(filters // 2, kernel_size=1, strides=2, use_bias=False)
        self.conv2 = Conv2D(filters // 2, kernel_size=1, strides=2, use_bias=False)
        self.bn = BatchNormalization(trainable=False)

    def call(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x[:, 1:, 1:, :])
        out = tf.concat([out1, out2], axis=3)
        out = self.bn(out)
        return out


class ENASLayer(MutableScope):
    def __init__(self, key, prev_labels, filters):
        super().__init__(key)
        self.mutable = LayerChoice([
            build_conv(filters, 3, 'conv3'),
            build_separable_conv(filters, 3, 'sepconv3'),
            build_conv(filters, 5, 'conv5'),
            build_separable_conv(filters, 5, 'sepconv5'),
            build_avg_pool(filters, 'avgpool'),
            build_max_pool(filters, 'maxpool'),
        ])
        if len(prev_labels) > 0:
            self.skipconnect = InputChoice(choose_from=prev_labels, n_chosen=None)
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
    def __init__(self, num_layers=12, filters=24, num_classes=10, dropout_rate=0.0):
        super().__init__()
        self.num_layers = num_layers

        self.stem = Sequential([
            Conv2D(filters, kernel_size=3, padding='same', use_bias=False),
            BatchNormalization()
        ])

        labels = ['layer_{}'.format(i) for i in range(num_layers)]
        self.enas_layers = []
        for i in range(num_layers):
            layer = ENASLayer(labels[i], labels[:i], filters)
            self.enas_layers.append(layer)

        pool_num = 2
        self.pool_distance = num_layers // (pool_num + 1)
        self.pool_layers = [FactorizedReduce(filters) for _ in range(pool_num)]

        self.gap = GlobalAveragePooling2D()
        self.dropout = Dropout(dropout_rate)
        self.dense = Dense(num_classes)

    def call(self, x):
        cur = self.stem(x)
        prev_outputs = [cur]

        for i, layer in enumerate(self.enas_layers):
            if i > 0 and i % self.pool_distance == 0:
                pool = self.pool_layers[i // self.pool_distance - 1]
                prev_outputs = [pool(tensor) for tensor in prev_outputs]
                cur = prev_outputs[-1]

            cur = layer(prev_outputs)
            prev_outputs.append(cur)

        cur = self.gap(cur)
        cur = self.dropout(cur)
        logits = self.dense(cur)
        return logits
