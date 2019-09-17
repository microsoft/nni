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


def conv_op(inputs, filter_size, is_training, count, out_filters,
                     data_format, ch_mul=1, start_idx=None, separable=False):
    """
    Args:
        start_idx: where to start taking the output channels. if None, assuming
            fixed_arc mode
        count: how many output_channels to take.
    """

    if data_format == "NHWC":
        inp_c = inputs.get_shape()[3].value
    elif data_format == "NCHW":
        inp_c = inputs.get_shape()[1].value

    with tf.variable_scope("inp_conv_1"):
        w = create_weight("w", [1, 1, inp_c, out_filters])
        x = tf.nn.conv2d(inputs, w, [1, 1, 1, 1],
                            "SAME", data_format=data_format)
        x = batch_norm(x, is_training, data_format=data_format)
        x = tf.nn.relu(x)

    with tf.variable_scope("out_conv_{}".format(filter_size)):
        if start_idx is None:
            if separable:
                w_depth = create_weight(
                    "w_depth", [filter_size, filter_size, out_filters, ch_mul])
                w_point = create_weight(
                    "w_point", [1, 1, out_filters * ch_mul, count])
                x = tf.nn.separable_conv2d(x, w_depth, w_point, strides=[1, 1, 1, 1],
                                            padding="SAME", data_format=data_format)
                x = batch_norm(
                    x, is_training, data_format=data_format)
            else:
                w = create_weight(
                    "w", [filter_size, filter_size, inp_c, count])
                x = tf.nn.conv2d(
                    x, w, [1, 1, 1, 1], "SAME", data_format=data_format)
                x = batch_norm(
                    x, is_training, data_format=data_format)
        else:
            if separable:
                w_depth = create_weight(
                    "w_depth", [filter_size, filter_size, out_filters, ch_mul])
                #test_depth = w_depth
                w_point = create_weight(
                    "w_point", [out_filters, out_filters * ch_mul])
                w_point = w_point[start_idx:start_idx+count, :]
                w_point = tf.transpose(w_point, [1, 0])
                w_point = tf.reshape(
                    w_point, [1, 1, out_filters * ch_mul, count])

                x = tf.nn.separable_conv2d(x, w_depth, w_point, strides=[1, 1, 1, 1],
                                            padding="SAME", data_format=data_format)
                mask = tf.range(0, out_filters, dtype=tf.int32)
                mask = tf.logical_and(
                    start_idx <= mask, mask < start_idx + count)
                x = batch_norm_with_mask(
                    x, is_training, mask, out_filters, data_format=data_format)
            else:
                w = create_weight(
                    "w", [filter_size, filter_size, out_filters, out_filters])
                w = tf.transpose(w, [3, 0, 1, 2])
                w = w[start_idx:start_idx+count, :, :, :]
                w = tf.transpose(w, [1, 2, 3, 0])
                x = tf.nn.conv2d(
                    x, w, [1, 1, 1, 1], "SAME", data_format=data_format)
                mask = tf.range(0, out_filters, dtype=tf.int32)
                mask = tf.logical_and(
                    start_idx <= mask, mask < start_idx + count)
                x = batch_norm_with_mask(
                    x, is_training, mask, out_filters, data_format=data_format)
        x = tf.nn.relu(x)
    return x

def pool_op(inputs, is_training, count, out_filters, avg_or_max, data_format, start_idx=None):
    """
    Args:
        start_idx: where to start taking the output channels. if None, assuming
            fixed_arc mode
        count: how many output_channels to take.
    """

    if data_format == "NHWC":
        inp_c = inputs.get_shape()[3].value
    elif data_format == "NCHW":
        inp_c = inputs.get_shape()[1].value

    with tf.variable_scope("conv_1"):
        w = create_weight("w", [1, 1, inp_c, out_filters])
        x = tf.nn.conv2d(inputs, w, [1, 1, 1, 1],
                            "SAME", data_format=data_format)
        x = batch_norm(x, is_training, data_format=data_format)
        x = tf.nn.relu(x)

    with tf.variable_scope("pool"):
        if data_format == "NHWC":
            actual_data_format = "channels_last"
        elif data_format == "NCHW":
            actual_data_format = "channels_first"

        if avg_or_max == "avg":
            x = tf.layers.average_pooling2d(
                x, [3, 3], [1, 1], "SAME", data_format=actual_data_format)
        elif avg_or_max == "max":
            x = tf.layers.max_pooling2d(
                x, [3, 3], [1, 1], "SAME", data_format=actual_data_format)
        else:
            raise ValueError("Unknown pool {}".format(avg_or_max))

        if start_idx is not None:
            if data_format == "NHWC":
                x = x[:, :, :, start_idx: start_idx+count]
            elif data_format == "NCHW":
                x = x[:, start_idx: start_idx+count, :, :]

    return x


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
