import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages


def lstm(x, prev_c, prev_h, w):
    ifog = tf.matmul(tf.concat([x, prev_h], axis=1), w)
    i, f, o, g = tf.split(ifog, 4, axis=1)
    i = tf.sigmoid(i)
    f = tf.sigmoid(f)
    o = tf.sigmoid(o)
    g = tf.tanh(g)
    next_c = i * g + f * prev_c
    next_h = o * tf.tanh(next_c)
    return next_c, next_h


def stack_lstm(x, prev_c, prev_h, w):
    next_c, next_h = [], []
    for layer_id, (_c, _h, _w) in enumerate(zip(prev_c, prev_h, w)):
        inputs = x if layer_id == 0 else next_h[-1]
        curr_c, curr_h = lstm(inputs, _c, _h, _w)
        next_c.append(curr_c)
        next_h.append(curr_h)
    return next_c, next_h


def create_weight(name, shape, initializer=None, trainable=True, seed=None):
    if initializer is None:
        initializer = tf.contrib.keras.initializers.he_normal(seed=seed)
    return tf.get_variable(name, shape, initializer=initializer, trainable=trainable)


def create_bias(name, shape, initializer=None):
    if initializer is None:
        initializer = tf.constant_initializer(0.0, dtype=tf.float32)
    return tf.get_variable(name, shape, initializer=initializer)


def global_avg_pool(x, data_format="NHWC"):
    if data_format == "NHWC":
        x = tf.reduce_mean(x, [1, 2])
    elif data_format == "NCHW":
        x = tf.reduce_mean(x, [2, 3])
    else:
        raise NotImplementedError("Unknown data_format {}".format(data_format))
    return x


def batch_norm(x, is_training, name="bn", decay=0.9, epsilon=1e-5,
               data_format="NHWC"):
    if data_format == "NHWC":
        shape = [x.get_shape()[3]]
    elif data_format == "NCHW":
        shape = [x.get_shape()[1]]
    else:
        raise NotImplementedError("Unknown data_format {}".format(data_format))

    with tf.variable_scope(name, reuse=None if is_training else True):
        offset = tf.get_variable(
            "offset", shape,
            initializer=tf.constant_initializer(0.0, dtype=tf.float32))
        scale = tf.get_variable(
            "scale", shape,
            initializer=tf.constant_initializer(1.0, dtype=tf.float32))
        moving_mean = tf.get_variable(
            "moving_mean", shape, trainable=False,
            initializer=tf.constant_initializer(0.0, dtype=tf.float32))
        moving_variance = tf.get_variable(
            "moving_variance", shape, trainable=False,
            initializer=tf.constant_initializer(1.0, dtype=tf.float32))

        if is_training:
            x, mean, variance = tf.nn.fused_batch_norm(
                x, scale, offset, epsilon=epsilon, data_format=data_format,
                is_training=True)
            update_mean = moving_averages.assign_moving_average(
                moving_mean, mean, decay)
            update_variance = moving_averages.assign_moving_average(
                moving_variance, variance, decay)
            with tf.control_dependencies([update_mean, update_variance]):
                x = tf.identity(x)
        else:
            x, _, _ = tf.nn.fused_batch_norm(x, scale, offset, mean=moving_mean,
                                             variance=moving_variance,
                                             epsilon=epsilon, data_format=data_format,
                                             is_training=False)
    return x


def batch_norm_with_mask(x, is_training, mask, num_channels, name="bn",
                         decay=0.9, epsilon=1e-3, data_format="NHWC"):

    shape = [num_channels]
    indices = tf.where(mask)
    indices = tf.to_int32(indices)
    indices = tf.reshape(indices, [-1])

    with tf.variable_scope(name, reuse=None if is_training else True):
        offset = tf.get_variable(
            "offset", shape,
            initializer=tf.constant_initializer(0.0, dtype=tf.float32))
        scale = tf.get_variable(
            "scale", shape,
            initializer=tf.constant_initializer(1.0, dtype=tf.float32))
        offset = tf.boolean_mask(offset, mask)
        scale = tf.boolean_mask(scale, mask)

        moving_mean = tf.get_variable(
            "moving_mean", shape, trainable=False,
            initializer=tf.constant_initializer(0.0, dtype=tf.float32))
        moving_variance = tf.get_variable(
            "moving_variance", shape, trainable=False,
            initializer=tf.constant_initializer(1.0, dtype=tf.float32))

        if is_training:
            x, mean, variance = tf.nn.fused_batch_norm(
                x, scale, offset, epsilon=epsilon, data_format=data_format,
                is_training=True)
            mean = (1.0 - decay) * (tf.boolean_mask(moving_mean, mask) - mean)
            variance = (1.0 - decay) * \
                (tf.boolean_mask(moving_variance, mask) - variance)
            update_mean = tf.scatter_sub(
                moving_mean, indices, mean, use_locking=True)
            update_variance = tf.scatter_sub(
                moving_variance, indices, variance, use_locking=True)
            with tf.control_dependencies([update_mean, update_variance]):
                x = tf.identity(x)
        else:
            masked_moving_mean = tf.boolean_mask(moving_mean, mask)
            masked_moving_variance = tf.boolean_mask(moving_variance, mask)
            x, _, _ = tf.nn.fused_batch_norm(x, scale, offset,
                                             mean=masked_moving_mean,
                                             variance=masked_moving_variance,
                                             epsilon=epsilon, data_format=data_format,
                                             is_training=False)
    return x
