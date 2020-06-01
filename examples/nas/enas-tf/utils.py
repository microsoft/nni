# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tensorflow as tf


def accuracy_metrics(y_true, logits):
    return {'enas_acc': accuracy(y_true, logits)}

def accuracy(y_true, logits):
    # y_true: shape=(batch_size) or (batch_size,1), type=integer
    # logits: shape=(batch_size, num_of_classes), type=float
    # returns float
    batch_size = y_true.shape[0]
    y_true = tf.squeeze(y_true)
    y_pred = tf.math.argmax(logits, axis=1)
    y_pred = tf.cast(y_pred, y_true.dtype)
    equal = tf.cast(y_pred == y_true, tf.int32)
    return tf.math.reduce_sum(equal).numpy() / batch_size
