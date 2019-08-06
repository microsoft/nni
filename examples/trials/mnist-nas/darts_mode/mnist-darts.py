"""A deep MNIST classifier using convolutional layers."""

import argparse
import logging
import math
import tempfile
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import operators as op

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
        Building network for mnist, meanwhile specifying its neural architecture search space
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

        """@nni.mutable_layers(
            {
                layer_choice: [op.conv2d(size=1, in_ch=1, out_ch=self.channel_1_num),
                               op.conv2d(size=3, in_ch=1, out_ch=self.channel_1_num),
                               op.twice_conv2d(size=3, in_ch=1, out_ch=self.channel_1_num),
                               op.twice_conv2d(size=7, in_ch=1, out_ch=self.channel_1_num),
                               op.dilated_conv(in_ch=1, out_ch=self.channel_1_num),
                               op.separable_conv(size=3, in_ch=1, out_ch=self.channel_1_num),
                               op.separable_conv(size=5, in_ch=1, out_ch=self.channel_1_num),
                               op.separable_conv(size=7, in_ch=1, out_ch=self.channel_1_num)],
                fixed_inputs: [x_image],
                layer_output: conv1_out
            },
            {
                layer_choice: [op.post_process(ch_size=self.channel_1_num)],
                fixed_inputs: [conv1_out],
                layer_output: post1_out
            },
            {
                layer_choice: [op.max_pool(size=3),
                               op.max_pool(size=5),
                               op.max_pool(size=7),
                               op.avg_pool(size=3),
                               op.avg_pool(size=5),
                               op.avg_pool(size=7)],
                fixed_inputs: [post1_out],
                layer_output: pool1_out
            },
            {
                layer_choice: [op.conv2d(size=1, in_ch=self.channel_1_num, out_ch=self.channel_2_num),
                               op.conv2d(size=3, in_ch=self.channel_1_num, out_ch=self.channel_2_num),
                               op.twice_conv2d(size=3, in_ch=self.channel_1_num, out_ch=self.channel_2_num),
                               op.twice_conv2d(size=7, in_ch=self.channel_1_num, out_ch=self.channel_2_num),
                               op.dilated_conv(in_ch=self.channel_1_num, out_ch=self.channel_2_num),
                               op.separable_conv(size=3, in_ch=self.channel_1_num, out_ch=self.channel_2_num),
                               op.separable_conv(size=5, in_ch=self.channel_1_num, out_ch=self.channel_2_num),
                               op.separable_conv(size=7, in_ch=self.channel_1_num, out_ch=self.channel_2_num)],
                fixed_inputs: [pool1_out],
                optional_inputs: [post1_out],
                optional_input_size: [0, 1],
                layer_output: conv2_out
            },
            {
                layer_choice: [op.post_process(ch_size=self.channel_2_num)],
                fixed_inputs: [conv2_out],
                layer_output: post2_out
            },
            {
                layer_choice: [op.max_pool(size=3),
                               op.max_pool(size=5),
                               op.max_pool(size=7),
                               op.avg_pool(size=3),
                               op.avg_pool(size=5),
                               op.avg_pool(size=7)],
                fixed_inputs: [post2_out],
                optional_inputs: [post1_out, pool1_out],
                optional_input_size: [0, 1],
                layer_output: pool2_out
            }
        )"""

        # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
        # is down to 7x7x64 feature maps -- maps this to 1024 features.
        last_dim_list = pool2_out.get_shape().as_list()
        assert(last_dim_list[1] == last_dim_list[2])
        last_dim = last_dim_list[1]
        with tf.name_scope('fc1'):
            w_fc1 = op.weight_variable(
                [last_dim * last_dim * self.channel_2_num, self.hidden_size])
            b_fc1 = op.bias_variable([self.hidden_size])

        h_pool2_flat = tf.reshape(
            pool2_out, [-1, last_dim * last_dim * self.channel_2_num])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

        # Dropout - controls the complexity of the model, prevents co-adaptation of features.
        with tf.name_scope('dropout'):
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # Map the 1024 features to 10 classes, one for each digit
        with tf.name_scope('fc2'):
            w_fc2 = op.weight_variable([self.hidden_size, self.y_dim])
            b_fc2 = op.bias_variable([self.y_dim])
            y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

        with tf.name_scope('loss'):
            self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=y_conv))
        with tf.name_scope('adam_optimizer'):
            self.train_step = tf.train.AdamOptimizer(
                self.learning_rate).minimize(self.cross_entropy)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(
                tf.argmax(y_conv, 1), tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32))


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
            feed_dict={mnist_network.images: batch[0],
                        mnist_network.labels: batch[1],
                        mnist_network.keep_prob: 1 - params['dropout_rate']}
            """@nni.training_update(tf, sess, mnist_network.cross_entropy)"""
            batch = mnist.train.next_batch(params['batch_size'])
            feed_dict={mnist_network.images: batch[0],
                        mnist_network.labels: batch[1],
                        mnist_network.keep_prob: 1 - params['dropout_rate']}
            mnist_network.train_step.run(feed_dict=feed_dict)

            if i % 100 == 0:
                test_acc = mnist_network.accuracy.eval(
                    feed_dict={mnist_network.images: mnist.test.images,
                               mnist_network.labels: mnist.test.labels,
                               mnist_network.keep_prob: 1.0})

                """@nni.report_intermediate_result(test_acc)"""
                logger.debug('test accuracy %g', test_acc)
                logger.debug('Pipe send intermediate result done.')

        test_acc = mnist_network.accuracy.eval(
            feed_dict={mnist_network.images: mnist.test.images,
                       mnist_network.labels: mnist.test.labels,
                       mnist_network.keep_prob: 1.0})

        """@nni.report_final_result(test_acc)"""
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
        params = vars(get_params())
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
