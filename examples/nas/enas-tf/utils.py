# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tensorflow as tf


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = tf.math.top_k(output, maxk)
    pred = tf.transpose(pred)
    # one-hot case
    if len(target.shape) > 1:
        target = tf.math.reduce_max(target, 1)[1]

    correct = (pred.reshape == target.reshape(pred.shape))
    correct = tf.reshape(tf.cast(correct, tf.float), [-1]).numpy()

    res = dict()
    for k in topk:
        correct_k = sum(correct[:k])
        res["acc{}".format(k)] = correct_k * (1.0 / batch_size)
    return res


def reward_accuracy(output, target, topk=(1,)):
    batch_size = target.shape[0]
    _, predicted = tf.math.reduce_max(output, axis=1)
    return tf.math.reduce_sum(tf.cast(predicted == target, tf.int32)).numpy() / batch_size
