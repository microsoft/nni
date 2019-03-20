import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages

from src.common_ops import create_weight
from src.common_ops import create_bias

def layer_norm(x, is_training, name="layer_norm"):
  x = tf.contrib.layers.layer_norm(x, scope=name,
                                   reuse=None if is_training else True)
  return x

def batch_norm(x, is_training, name="batch_norm", decay=0.999, epsilon=1.0):
  shape = x.get_shape()[1]
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
      mean, variance = tf.nn.moments(x, [0])
      update_mean = moving_averages.assign_moving_average(
        moving_mean, mean, decay)
      update_variance = moving_averages.assign_moving_average(
        moving_variance, variance, decay)

      with tf.control_dependencies([update_mean, update_variance]):
        x = scale * (x - mean) / tf.sqrt(epsilon + variance) + offset
    else:
      x = scale * (x - moving_mean) / tf.sqrt(epsilon + moving_variance) + offset
  return x


