"""A deep MNIST classifier using convolutional layers."""

import argparse
import logging
import math
import tempfile
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import nni

FLAGS = None

logger = logging.getLogger('mnist_AutoML')


class MnistNetwork(object):
    '''
    MnistNetwork is for initializing and building basic network for mnist.
    '''
    def __init__(self,
                 channel_1_num,
                 channel_2_num,
                 conv_size,
                 hidden_size,
                 pool_size,
                 learning_rate,
                 x_dim=784,
                 y_dim=10):
        self.channel_1_num = channel_1_num
        self.channel_2_num = channel_2_num
        self.conv_size = conv_size
        self.hidden_size = hidden_size
        self.pool_size = pool_size
        self.learning_rate = learning_rate
        self.x_dim = x_dim
        self.y_dim = y_dim

        self.images = tf.placeholder(tf.float32, [None, self.x_dim], name='input_x')
        self.labels = tf.placeholder(tf.float32, [None, self.y_dim], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.train_step = None
        self.accuracy = None

    def build_network(self):
        '''
        Building network for mnist
        '''

        # Reshape to use within a convolutional neural net.
        # Last dimension is for "features" - there is only one here, since images are
        # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
        with tf.name_scope('reshape'):
            try:
                input_dim = int(math.sqrt(self.x_dim))
            except:
                print(
                    'input dim cannot be sqrt and reshape. input dim: ' + str(self.x_dim))
                logger.debug(
                    'input dim cannot be sqrt and reshape. input dim: %s', str(self.x_dim))
                raise
            x_image = tf.reshape(self.images, [-1, input_dim, input_dim, 1])

        # First convolutional layer - maps one grayscale image to 32 feature maps.
        with tf.name_scope('conv1'):
            w_conv1 = weight_variable(
                [self.conv_size, self.conv_size, 1, self.channel_1_num])
            b_conv1 = bias_variable([self.channel_1_num])
            h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)

        # Pooling layer - downsamples by 2X.
        with tf.name_scope('pool1'):
            h_pool1 = max_pool(h_conv1, self.pool_size)

        # Second convolutional layer -- maps 32 feature maps to 64.
        with tf.name_scope('conv2'):
            w_conv2 = weight_variable([self.conv_size, self.conv_size,
                                       self.channel_1_num, self.channel_2_num])
            b_conv2 = bias_variable([self.channel_2_num])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)

        # Second pooling layer.
        with tf.name_scope('pool2'):
            h_pool2 = max_pool(h_conv2, self.pool_size)

        # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
        # is down to 7x7x64 feature maps -- maps this to 1024 features.
        last_dim = int(input_dim / (self.pool_size * self.pool_size))
        with tf.name_scope('fc1'):
            w_fc1 = weight_variable(
                [last_dim * last_dim * self.channel_2_num, self.hidden_size])
            b_fc1 = bias_variable([self.hidden_size])

        h_pool2_flat = tf.reshape(
            h_pool2, [-1, last_dim * last_dim * self.channel_2_num])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

        # Dropout - controls the complexity of the model, prevents co-adaptation of features.
        with tf.name_scope('dropout'):
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # Map the 1024 features to 10 classes, one for each digit
        with tf.name_scope('fc2'):
            w_fc2 = weight_variable([self.hidden_size, self.y_dim])
            b_fc2 = bias_variable([self.y_dim])
            y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

        with tf.name_scope('loss'):
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=y_conv))
        with tf.name_scope('adam_optimizer'):
            self.train_step = tf.train.AdamOptimizer(
                self.learning_rate).minimize(cross_entropy)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(
                tf.argmax(y_conv, 1), tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32))


def conv2d(x_input, w_matrix):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x_input, w_matrix, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x_input, pool_size):
    """max_pool downsamples a feature map by 2X."""
    return tf.nn.max_pool(x_input, ksize=[1, pool_size, pool_size, 1],
                          strides=[1, pool_size, pool_size, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def download_mnist_retry(data_dir, max_num_retries=20):
    """Try to download mnist dataset and avoid errors"""
    for _ in range(max_num_retries):
        try:
            return input_data.read_data_sets(data_dir, one_hot=True)
        except tf.errors.AlreadyExistsError:
            time.sleep(1)
    raise Exception("Failed to download MNIST.")

def main(params):
    '''
    Main function, build mnist network, run and send result to NNI.
    '''
    # Import data
    mnist = download_mnist_retry(params['data_dir'])
    print('Mnist download data done.')
    logger.debug('Mnist download data done.')

    # Create the model
    # Build the graph for the deep net
    mnist_network = MnistNetwork(channel_1_num=params['channel_1_num'],
                                 channel_2_num=params['channel_2_num'],
                                 conv_size=params['conv_size'],
                                 hidden_size=params['hidden_size'],
                                 pool_size=params['pool_size'],
                                 learning_rate=params['learning_rate'])
    mnist_network.build_network()
    logger.debug('Mnist build network done.')

    # Write log
    graph_location = tempfile.mkdtemp()
    logger.debug('Saving graph to: %s', graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    test_acc = 0.0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(params['batch_num']):
            batch = mnist.train.next_batch(params['batch_size'])
            mnist_network.train_step.run(feed_dict={mnist_network.images: batch[0],
                                                    mnist_network.labels: batch[1],
                                                    mnist_network.keep_prob: 1 - params['dropout_rate']}
                                        )

            if i % 100 == 0:
                test_acc = mnist_network.accuracy.eval(
                    feed_dict={mnist_network.images: mnist.test.images,
                               mnist_network.labels: mnist.test.labels,
                               mnist_network.keep_prob: 1.0})

                nni.report_intermediate_result(test_acc)
                logger.debug('test accuracy %g', test_acc)
                logger.debug('Pipe send intermediate result done.')

        test_acc = mnist_network.accuracy.eval(
            feed_dict={mnist_network.images: mnist.test.images,
                       mnist_network.labels: mnist.test.labels,
                       mnist_network.keep_prob: 1.0})

        nni.report_final_result(test_acc)
        logger.debug('Final result is %g', test_acc)
        logger.debug('Send final result done.')

def get_params():
    ''' Get parameters from command line '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='/tmp/tensorflow/mnist/input_data', help="data directory")
    parser.add_argument("--dropout_rate", type=float, default=0.5, help="dropout rate")
    parser.add_argument("--channel_1_num", type=int, default=32)
    parser.add_argument("--channel_2_num", type=int, default=64)
    parser.add_argument("--conv_size", type=int, default=5)
    parser.add_argument("--pool_size", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_num", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)

    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(get_params())
        params.update(tuner_params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
