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


def build_conv_1x1(filters, name=None):
    return Sequential([
        Conv2D(filters, kernel_size=1, use_bias=False),
        BatchNormalization(trainable=False),
        ReLU(),
    ], name)

def build_sep_conv(filters, kernel_size, name=None):
    return Sequential([
        ReLU(),
        SeparableConv2D(filters, kernel_size, padding='same'),
        BatchNormalization(trainable=True),
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


class ReductionLayer(Model):
    def __init__(self, filters):
        super().__init__()
        self.reduce0 = FactorizedReduce(filters)
        self.reduce1 = FactorizedReduce(filters)

    def call(self, prevprev, prev):
        return self.reduce0(prevprev), self.reduce1(prev)


class Calibration(Model):
    def __init__(self, filters):
        super().__init__()
        self.filters = filters
        self.process = None

    def build(self, shape):
        assert len(shape) == 4  # batch_size, width, height, filters
        if shape[3] != self.filters:
            self.process = build_conv_1x1(self.filters)

    def call(self, x):
        if self.process is None:
            return x
        return self.process(x)


class Cell(Model):
    def __init__(self, cell_name, prev_labels, filters):
        super().__init__()
        self.input_choice = InputChoice(choose_from=prev_labels, n_chosen=1, return_mask=True, key=cell_name + '_input')
        self.op_choice = LayerChoice([
            build_sep_conv(filters, 3),
            build_sep_conv(filters, 5),
            AveragePooling2D(pool_size=3, strides=1, padding='same'),
            MaxPool2D(pool_size=3, strides=1, padding='same'),
            Sequential(),  # Identity
        ], key=cell_name + '_op')

    def call(self, prev_layers):
        chosen_input, chosen_mask = self.input_choice(prev_layers)
        cell_out = self.op_choice(chosen_input)
        return cell_out, chosen_mask


class Node(MutableScope):
    def __init__(self, node_name, prev_node_names, filters):
        super().__init__(node_name)
        self.cell_x = Cell(node_name + '_x', prev_node_names, filters)
        self.cell_y = Cell(node_name + '_y', prev_node_names, filters)

    def call(self, prev_layers):
        out_x, mask_x = self.cell_x(prev_layers)
        out_y, mask_y = self.cell_y(prev_layers)
        return out_x + out_y, mask_x | mask_y


class ENASLayer(Model):
    def __init__(self, num_nodes, filters, reduction):
        super().__init__()
        self.preproc0 = Calibration(filters)
        self.preproc1 = Calibration(filters)

        self.nodes = []
        node_labels = [InputChoice.NO_KEY, InputChoice.NO_KEY]
        name_prefix = 'reduce' if reduction else 'normal'
        for i in range(num_nodes):
            node_labels.append('{}_node_{}'.format(name_prefix, i))
            self.nodes.append(Node(node_labels[-1], node_labels[:-1], filters))

        self.conv_ops = [Conv2D(filters, kernel_size=1, padding='same', use_bias=False) for _ in range(num_nodes + 2)]
        self.bn = BatchNormalization(trainable=False)

    def call(self, prevprev, prev):
        prev_nodes_out = [self.preproc0(prevprev), self.preproc1(prev)]
        nodes_used_mask = tf.zeros(len(self.nodes) + 2, dtype=tf.bool)
        for i, node in enumerate(self.nodes):
            node_out, mask = node(prev_nodes_out)
            nodes_used_mask |= tf.pad(mask, [[0, nodes_used_mask.shape[0] - mask.shape[0]]])
            prev_nodes_out.append(node_out)

        outputs = []
        for used, out, conv in zip(nodes_used_mask.numpy(), prev_nodes_out, self.conv_ops):
            if not used:
                outputs.append(conv(out))
        out = tf.add_n(outputs)
        return prev, self.bn(out)


class MicroNetwork(Model):
    def __init__(self, num_layers=6, num_nodes=5, out_channels=20, num_classes=10, dropout_rate=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.stem = Sequential([
            Conv2D(out_channels * 3, kernel_size=3, padding='same', use_bias=False),
            BatchNormalization(),
        ])

        pool_distance = num_layers // 3
        pool_layer_indices = [pool_distance, 2 * pool_distance + 1]

        self.enas_layers = []

        filters = out_channels
        for i in range(num_layers + 2):
            if i in pool_layer_indices:
                reduction = True
                filters *= 2
                self.enas_layers.append(ReductionLayer(filters))
            else:
                reduction = False
            self.enas_layers.append(ENASLayer(num_nodes, filters, reduction))

        self.gap = GlobalAveragePooling2D()
        self.dropout = Dropout(dropout_rate)
        self.dense = Dense(num_classes)

    def call(self, x):
        prev = cur = self.stem(x)
        for layer in self.enas_layers:
            prev, cur = layer(prev, cur)
        cur = tf.keras.activations.relu(cur)
        cur = self.gap(cur)
        cur = self.dropout(cur)
        logits = self.dense(cur)
        return logits
