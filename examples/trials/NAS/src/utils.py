from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf


user_flags = []


def DEFINE_string(name, default_value, doc_string):
    tf.app.flags.DEFINE_string(name, default_value, doc_string)
    global user_flags
    user_flags.append(name)


def DEFINE_integer(name, default_value, doc_string):
    tf.app.flags.DEFINE_integer(name, default_value, doc_string)
    global user_flags
    user_flags.append(name)


def DEFINE_float(name, default_value, doc_string):
    tf.app.flags.DEFINE_float(name, default_value, doc_string)
    global user_flags
    user_flags.append(name)


def DEFINE_boolean(name, default_value, doc_string):
    tf.app.flags.DEFINE_boolean(name, default_value, doc_string)
    global user_flags
    user_flags.append(name)


def print_user_flags(line_limit=80):
    print("-" * 80)

    global user_flags
    FLAGS = tf.app.flags.FLAGS

    for flag_name in sorted(user_flags):
        value = "{}".format(getattr(FLAGS, flag_name))
        log_string = flag_name
        log_string += "." * (line_limit - len(flag_name) - len(value))
        log_string += value
        print(log_string)


def get_C(x, data_format):
    """
    Args:
        x: tensor of shape [N, H, W, C] or [N, C, H, W]
    """
    if data_format == "NHWC":
        return x.get_shape()[3].value
    elif data_format == "NCHW":
        return x.get_shape()[1].value
    else:
        raise ValueError(
            "Unknown data_format '{0}'".format(data_format))

def get_HW(x, data_format):
    """
    Args:
        x: tensor of shape [N, H, W, C] or [N, C, H, W]
    """
    return x.get_shape()[2].value

def get_strides(stride, data_format):
    """
    Args:
        x: tensor of shape [N, H, W, C] or [N, C, H, W]
    """
    if data_format == "NHWC":
        return [1, stride, stride, 1]
    elif data_format == "NCHW":
        return [1, 1, stride, stride]
    else:
        raise ValueError(
            "Unknown data_format '{0}'".format(data_format))


class TextColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Logger(object):
    def __init__(self, output_file):
        self.terminal = sys.stdout
        self.log = open(output_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()


def count_model_params(tf_variables):
    """
    Args:
        tf_variables: list of all model variables
    """

    num_vars = 0
    for var in tf_variables:
        num_vars += np.prod([dim.value for dim in var.get_shape()])
    return num_vars


def get_train_ops(
        loss,
        tf_variables,
        train_step,
        clip_mode=None,
        grad_bound=None,
        l2_reg=1e-4,
        lr_warmup_val=None,
        lr_warmup_steps=100,
        lr_init=0.1,
        lr_dec_start=0,
        lr_dec_every=10000,
        lr_dec_rate=0.1,
        lr_dec_min=None,
        lr_cosine=False,
        lr_max=None,
        lr_min=None,
        lr_T_0=None,
        lr_T_mul=None,
        num_train_batches=None,
        optim_algo=None,
        sync_replicas=False,
        num_aggregate=None,
        num_replicas=None,
        get_grad_norms=False,
        moving_average=None):
    """
    Args:
        clip_mode: "global", "norm", or None.
        moving_average: store the moving average of parameters
    """

    if l2_reg > 0:
        l2_losses = []
        for var in tf_variables:
            l2_losses.append(tf.reduce_sum(var ** 2))
        l2_loss = tf.add_n(l2_losses)
        loss += l2_reg * l2_loss

    grads = tf.gradients(loss, tf_variables)
    grad_norm = tf.global_norm(grads)

    grad_norms = {}
    for v, g in zip(tf_variables, grads):
        if v is None or g is None:
            continue
        if isinstance(g, tf.IndexedSlices):
            grad_norms[v.name] = tf.sqrt(tf.reduce_sum(g.values ** 2))
        else:
            grad_norms[v.name] = tf.sqrt(tf.reduce_sum(g ** 2))

    if clip_mode is not None:
        assert grad_bound is not None, "Need grad_bound to clip gradients."
        if clip_mode == "global":
            grads, _ = tf.clip_by_global_norm(grads, grad_bound)
        elif clip_mode == "norm":
            clipped = []
            for g in grads:
                if isinstance(g, tf.IndexedSlices):
                    c_g = tf.clip_by_norm(g.values, grad_bound)
                    c_g = tf.IndexedSlices(g.indices, c_g)
                else:
                    c_g = tf.clip_by_norm(g, grad_bound)
                clipped.append(g)
            grads = clipped
        else:
            raise NotImplementedError("Unknown clip_mode {}".format(clip_mode))

    if lr_cosine:
        assert lr_max is not None, "Need lr_max to use lr_cosine"
        assert lr_min is not None, "Need lr_min to use lr_cosine"
        assert lr_T_0 is not None, "Need lr_T_0 to use lr_cosine"
        assert lr_T_mul is not None, "Need lr_T_mul to use lr_cosine"
        assert num_train_batches is not None, ("Need num_train_batches to use"
                                               " lr_cosine")

        curr_epoch = train_step // num_train_batches

        last_reset = tf.Variable(0, dtype=tf.int32, trainable=False,
                                 name="last_reset")
        T_i = tf.Variable(lr_T_0, dtype=tf.int32, trainable=False, name="T_i")
        T_curr = curr_epoch - last_reset

        def _update():
            update_last_reset = tf.assign(
                last_reset, curr_epoch, use_locking=True)
            update_T_i = tf.assign(T_i, T_i * lr_T_mul, use_locking=True)
            with tf.control_dependencies([update_last_reset, update_T_i]):
                rate = tf.to_float(T_curr) / tf.to_float(T_i) * 3.1415926
                lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + tf.cos(rate))
            return lr

        def _no_update():
            rate = tf.to_float(T_curr) / tf.to_float(T_i) * 3.1415926
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + tf.cos(rate))
            return lr

        learning_rate = tf.cond(
            tf.greater_equal(T_curr, T_i), _update, _no_update)
    else:
        learning_rate = tf.train.exponential_decay(
            lr_init, tf.maximum(train_step - lr_dec_start, 0), lr_dec_every,
            lr_dec_rate, staircase=True)
        if lr_dec_min is not None:
            learning_rate = tf.maximum(learning_rate, lr_dec_min)

    if lr_warmup_val is not None:
        learning_rate = tf.cond(tf.less(train_step, lr_warmup_steps),
                                lambda: lr_warmup_val, lambda: learning_rate)

    if optim_algo == "momentum":
        opt = tf.train.MomentumOptimizer(
            learning_rate, 0.9, use_locking=True, use_nesterov=True)
    elif optim_algo == "sgd":
        opt = tf.train.GradientDescentOptimizer(
            learning_rate, use_locking=True)
    elif optim_algo == "adam":
        opt = tf.train.AdamOptimizer(learning_rate, beta1=0.0, epsilon=1e-3,
                                     use_locking=True)
    else:
        raise ValueError("Unknown optim_algo {}".format(optim_algo))

    if sync_replicas:
        assert num_aggregate is not None, "Need num_aggregate to sync."
        assert num_replicas is not None, "Need num_replicas to sync."

        opt = tf.train.SyncReplicasOptimizer(
            opt,
            replicas_to_aggregate=num_aggregate,
            total_num_replicas=num_replicas,
            use_locking=True)

    if moving_average is not None:
        opt = tf.contrib.opt.MovingAverageOptimizer(
            opt, average_decay=moving_average)

    train_op = opt.apply_gradients(
        zip(grads, tf_variables), global_step=train_step)

    if get_grad_norms:
        return train_op, learning_rate, grad_norm, opt, grad_norms
    else:
        return train_op, learning_rate, grad_norm, opt
