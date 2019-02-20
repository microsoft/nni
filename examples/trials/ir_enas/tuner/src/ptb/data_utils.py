import sys
import numpy as np
import tensorflow as tf


def ptb_input_producer(raw_data, batch_size, num_steps, shuffle=False,
                       randomize=False):
  """
  Args:
    raw_data: np tensor of size [num_words].
    batch_size: self-explained.
    num_steps: number of BPTT steps.
  """

  num_batches_per_epoch = ((np.size(raw_data) // batch_size) - 1) // num_steps
  raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

  data_len = tf.size(raw_data)
  batch_len = data_len // batch_size
  data = tf.reshape(raw_data[0 : batch_size * batch_len],
                    [batch_size, batch_len])

  epoch_size = (batch_len - 1) // num_steps
  with tf.device("/cpu:0"):
    epoch_size = tf.identity(epoch_size, name="epoch_size")
    
    if randomize:
      i = tf.random_uniform([1], minval=0, maxval=batch_len - num_steps,
                            dtype=tf.int32)
      i = tf.reduce_sum(i)
      x = tf.strided_slice(
        data, [0, i], [batch_size, i + num_steps])
      y = tf.strided_slice(
        data, [0, i + 1], [batch_size, i + num_steps + 1])
    else:
      i = tf.train.range_input_producer(epoch_size, shuffle=shuffle).dequeue()
      x = tf.strided_slice(
        data, [0, i * num_steps], [batch_size, (i + 1) * num_steps])
      y = tf.strided_slice(
        data, [0, i * num_steps + 1], [batch_size, (i + 1) * num_steps + 1])

    x.set_shape([batch_size, num_steps])
    y.set_shape([batch_size, num_steps])

  return x, y, num_batches_per_epoch

