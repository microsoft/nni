# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tensorflow as tf

def get_dataset():
    (x_train, y_train), (x_valid, y_valid) = tf.keras.datasets.cifar10.load_data()
    x_train, x_valid = x_train / 255.0, x_valid / 255.0
    train_set = (x_train, y_train)
    valid_set = (x_valid, y_valid)
    return train_set, valid_set
