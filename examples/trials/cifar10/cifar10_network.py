# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge,
# to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

'''
Building the cifar10 network, run and send result to NNI.
'''

import logging

import tensorflow as tf
import nni
import cifar10

_logger = logging.getLogger("cifar10_automl")

NUM_CLASS = 10
MAX_BATCH_NUM = 5000
#MAX_BATCH_NUM = 50


def activation_functions(act):
    '''
    Choose activation function by index
    '''
    if act == 1:
        return tf.nn.softmax
    if act == 2:
        return tf.nn.tanh
    if act == 3:
        return tf.nn.relu
    if act == 4:
        return tf.nn.relu
    if act == 5:
        return tf.nn.elu
    if act == 6:
        return tf.nn.leaky_relu
    return None


def get_optimizer(opt):
    '''
    Return optimizer by index
    '''
    if opt == 1:
        return tf.train.GradientDescentOptimizer
    if opt == 2:
        return tf.train.RMSPropOptimizer
    if opt == 3:
        return tf.train.AdagradOptimizer
    if opt == 4:
        return tf.train.AdadeltaOptimizer
    if opt == 5:
        return tf.train.AdamOptimizer
    assert False
    return None


class Cifar10(object):
    '''
    Class Cifar10 could build and run network for cifar10.
    '''
    def __init__(self):
        # Place holder
        self.is_train = tf.placeholder('int32')
        self.keep_prob1 = tf.placeholder('float', name='xa')
        self.keep_prob2 = tf.placeholder('float', name='xb')

        self.accuracy = None
        self.train_op = None

    def build_network(self, config):
        """
        Build network for CIFAR-10 and train.
        """
        num_classes = NUM_CLASS
        batch_size = config['batch_size']
        num_units = config['conv_units_size']
        conv_size = config['conv_size']
        num_blocks = config['num_blocks']
        initial_method = config['initial_method']
        act_notlast = config['act_notlast']
        pool_size = config['pool_size']
        hidden_size = config['hidden_size']
        act = config['act']
        learning_rate = config['learning_rate']
        opt = get_optimizer(config['optimizer'])

        is_train = self.is_train
        keep_prob1 = self.keep_prob1
        keep_prob2 = self.keep_prob2

        # Get images and labels for CIFAR-10.
        with tf.device('/cpu:0'):
            images, labels = cifar10.distorted_inputs()
            test_images, test_labels = cifar10.inputs('test')

        # Choose test set or train set by is_train
        images = images * tf.cast(is_train, tf.float32) + \
            (1-tf.cast(is_train, tf.float32)) * test_images
        labels = labels * is_train + (1 - is_train) * test_labels

        input_vec = tf.slice(images, [0, 0, 0, 0], [batch_size, 24, 24, 3])
        output = tf.slice(labels, [0], [batch_size])
        output = tf.one_hot(output, num_classes)

        input_units = 3
        for num in range(num_blocks):
            if initial_method == 1:
                conv_layer = tf.Variable(tf.truncated_normal(shape=[conv_size, conv_size,
                                                                    input_units, num_units],
                                                             stddev=1.0 / num_units))
            else:
                conv_layer = tf.Variable(tf.random_uniform(shape=[conv_size, conv_size,
                                                                  input_units, num_units],
                                                           minval=-0.05, maxval=0.05))
            input_units = num_units
            input_vec = tf.nn.conv2d(input_vec, conv_layer, strides=[1, 1, 1, 1], padding='SAME')
            act_no_f = activation_functions(act_notlast)
            input_vec = act_no_f(input_vec)

            input_vec = tf.layers.batch_normalization(input_vec)

            input_vec = tf.nn.dropout(input_vec, keep_prob=keep_prob1)

            if num >= num_blocks - 2:
                input_vec = tf.nn.max_pool(input_vec, ksize=[1, pool_size, pool_size, 1],
                                           strides=[1, 2, 2, 1], padding='SAME')
                num_units = num_units * 2

        input_vec = tf.contrib.layers.flatten(input_vec)

        input_vec = tf.layers.dense(
            input_vec, hidden_size, activation=activation_functions(act))

        input_vec = tf.layers.batch_normalization(input_vec)

        input_vec = tf.nn.dropout(input_vec, keep_prob=keep_prob2)

        input_vec = tf.layers.dense(input_vec, num_classes)

        logit = tf.nn.softmax_cross_entropy_with_logits(
            logits=input_vec, labels=output)

        loss = tf.reduce_mean(logit)

        accuracy = tf.equal(tf.argmax(input_vec, 1), tf.argmax(output, 1))
        self.accuracy = tf.reduce_mean(
            tf.cast(accuracy, "float"))  # add a reduce_mean
        self.train_op = opt(learning_rate=learning_rate).minimize(loss)

    def train(self, config):
        """
        train the cifar10 network
        """
        _logger.debug('Config is: %s', str(config))
        assert config['batch_size']
        assert config['conv_units_size']
        assert config['conv_size']
        assert config['num_blocks']
        assert config['initial_method']
        assert config['act_notlast']
        assert config['pool_size']
        assert config['hidden_size']
        assert config['act']
        assert config['dropout']
        assert config['learning_rate']
        assert config['optimizer']

        self.build_network(config)

        with tf.Session() as sess:
            # Initialize variables
            tf.initialize_all_variables().run()
            _logger.debug('Initialize all variables done.')

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)
            cnt = 0
            for cnt in range(MAX_BATCH_NUM):
                cnt = cnt + 1
                if cnt % 2000 == 0:
                    _logger.debug('Runing in batch %s', str(cnt))
                    acc = sess.run(self.accuracy, feed_dict={self.is_train: 0,
                                                                self.keep_prob1: 1.0,
                                                                self.keep_prob2: 1.0})
                    # Send intermediate result
                    nni.report_intermediate_result(acc)
                    _logger.debug('Report intermediate result done.')

                sess.run(self.train_op, feed_dict={self.is_train: 1,
                                                    self.keep_prob1: 1 - config['dropout'],
                                                    self.keep_prob2: config['dropout']})

            coord.request_stop()
            coord.join(threads)
        # Send final result
        nni.report_final_result(acc)
        _logger.debug('Training cifar10 done.')


def get_default_params():
    '''
    Return default parameters.
    '''
    config = {}
    config['learning_rate'] = 0.1
    config['batch_size'] = 512
    config['num_epochs'] = 100
    config['dropout'] = 0.5
    config['hidden_size'] = 1682
    config['conv_size'] = 5
    config['num_blocks'] = 3
    config['conv_units_size'] = 32
    config['pool_size'] = 3
    config['act_notlast'] = 5
    config['act'] = 2
    config['optimizer'] = 5
    config['initial_method'] = 2
    return config


if __name__ == '__main__':
    try:
        RCV_CONFIG = nni.get_parameters()
        _logger.debug(RCV_CONFIG)
        cifar10.maybe_download_and_extract()
        train_cifar10 = Cifar10()
        params = get_default_params()
        params.update(RCV_CONFIG)
        train_cifar10.train(params)
    except Exception as exception:
        _logger.exception(exception)
        raise
