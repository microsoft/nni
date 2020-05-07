# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tensorflow as tf


def get_dataset(batch_size):
    train_set, test_set = tf.keras.datasets.cifar10.load_data()
    train_x, train_y = train_set

    split = len(train_x) * 0.9
    enas_train_set = tf.data.Dataset.from_tensor_slices((train_x[:split], train_y[:split])).batch(batch_size)
    enas_test_set = tf.data.Dataset.from_tensor_slices((train_x[split:], train_y[split:])).batch(batch_size)

    enas_valid_set = tf.Data.Dataset.from_tensor_slices(test_set).batch(batch_size)

    return enas_train_set, enas_test_set, enas_valid_set
