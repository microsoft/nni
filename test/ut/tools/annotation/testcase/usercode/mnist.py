# -*- encoding:utf8 -*-

"""A deep MNIST classifier using convolutional layers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import tempfile
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

logger = logging.getLogger('mnist')

FLAGS = None

class MnistNetwork(object):
    def __init__(self,
                 channel_1_num = 32,
                 channel_2_num = 64,
                 conv_size = 5,
                 hidden_size = 1024,
                 pool_size = 2,
                 learning_rate = 0.0001,
                 x_dim = 784,
                 y_dim = 10):
        self.channel_1_num = channel_1_num
        self.channel_2_num = channel_2_num
        '''@nni.variable(nni.choice(2,3,5,7),name=self.conv_size)'''
        self.conv_size = conv_size
        '''@nni.variable(nni.choice(124,512,1024),name=self.hidden_size)'''
        self.hidden_size = hidden_size
        self.pool_size = pool_size
        '''@nni.variable(nni.randint(2,3,5),name=self.learning_rate)'''
        self.learning_rate = learning_rate
        self.x_dim = x_dim
        self.y_dim = y_dim

    def build_network(self):
        self.x = tf.placeholder(tf.float32, [None, self.x_dim], name = 'input_x')
        self.y = tf.placeholder(tf.float32, [None, self.y_dim], name = 'input_y')
        self.keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

        # Reshape to use within a convolutional neural net.
        # Last dimension is for "features" - there is only one here, since images are
        # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
        with tf.name_scope('reshape'):
            try:
                input_dim = int(math.sqrt(self.x_dim))
            except:
                #print('input dim cannot be sqrt and reshape. input dim: ' + str(self.x_dim))
                logger.debug('input dim cannot be sqrt and reshape. input dim: ', str(self.x_dim))
                raise
            x_image = tf.reshape(self.x, [-1, input_dim, input_dim, 1])

        # First convolutional layer - maps one grayscale image to 32 feature maps.
        with tf.name_scope('conv1'):
            W_conv1 = weight_variable([self.conv_size, self.conv_size, 1, self.channel_1_num])
            b_conv1 = bias_variable([self.channel_1_num])
            """@nni.function_choice(tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1),tf.nn.sigmoid(conv2d(x_image, W_conv1) + b_conv1),tf.nn.tanh(conv2d(x_image, W_conv1) + b_conv1),name=tf.nn.relu)"""
            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

        # Pooling layer - downsamples by 2X.
        with tf.name_scope('pool1'):
            """@nni.function_choice(max_pool(h_conv1, self.pool_size),avg_pool(h_conv1, self.pool_size),name=max_pool)"""
            h_pool1 = max_pool(h_conv1, self.pool_size)

        # Second convolutional layer -- maps 32 feature maps to 64.
        with tf.name_scope('conv2'):
            W_conv2 = weight_variable([self.conv_size, self.conv_size, self.channel_1_num, self.channel_2_num])
            b_conv2 = bias_variable([self.channel_2_num])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

        # Second pooling layer.
        with tf.name_scope('pool2'):
            #"""@nni.dynamic(input={cnn_block:1, concat:2},function_choice={"cnn_block":(x,nni.choice([3,4])),"cnn_block":(x),"concat":(x,y)},limit={"cnn_block.input":[concat,input],"concat.input":[this.depth-1,this.depth-3,this.depth-5],"graph.width":[1]})"""
            h_pool2 = max_pool(h_conv2, self.pool_size)

        # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
        # is down to 7x7x64 feature maps -- maps this to 1024 features.
        last_dim = int(input_dim / (self.pool_size * self.pool_size))
        with tf.name_scope('fc1'):
            W_fc1 = weight_variable([last_dim * last_dim * self.channel_2_num, self.hidden_size])
            b_fc1 = bias_variable([self.hidden_size])

        h_pool2_flat = tf.reshape(h_pool2, [-1, last_dim * last_dim * self.channel_2_num])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # Dropout - controls the complexity of the model, prevents co-adaptation of features.
        with tf.name_scope('dropout'):
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # Map the 1024 features to 10 classes, one for each digit
        with tf.name_scope('fc2'):
            W_fc2 = weight_variable([self.hidden_size, self.y_dim])
            b_fc2 = bias_variable([self.y_dim])
            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        with tf.name_scope('loss'):
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.y, logits = y_conv))
        with tf.name_scope('adam_optimizer'):
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return

def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x, pool_size):
    """max_pool downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, pool_size, pool_size, 1],
                        strides=[1, pool_size, pool_size, 1], padding='SAME')
def avg_pool(x,pool_size):
    return tf.nn.avg_pool(x, ksize=[1, pool_size, pool_size, 1],
                        strides=[1, pool_size, pool_size, 1], padding='SAME')

def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def main():
    # Import data
    data_dir= '/tmp/tensorflow/mnist/input_data'
    mnist = input_data.read_data_sets(data_dir, one_hot=True)
    logger.debug('Mnist download data done.')

    # Create the model
    # Build the graph for the deep net
    mnist_network = MnistNetwork()
    mnist_network.build_network()
    logger.debug('Mnist build network done.')

    # Write log
    graph_location = tempfile.mkdtemp()
    logger.debug('Saving graph to: %s', graph_location)
    # print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    test_acc = 0.0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        batch_num=200
        for i in range(batch_num):
            '''@nni.variable(nni.choice(50,250,500),name=batch_size)'''
            batch_size=50
            batch = mnist.train.next_batch(batch_size)
            '''@nni.variable(nni.choice(1,5),name=dropout_rate)'''
            dropout_rate=0.5
            mnist_network.train_step.run(feed_dict={mnist_network.x: batch[0], mnist_network.y: batch[1], mnist_network.keep_prob: dropout_rate})

            if i % 100 == 0:
                #train_accuracy = mnist_network.accuracy.eval(feed_dict={
                #    mnist_network.x: batch[0], mnist_network.y: batch[1], mnist_network.keep_prob: params['dropout_rate']})
                #print('step %d, training accuracy %g' % (i, train_accuracy))

                test_acc = mnist_network.accuracy.eval(feed_dict={
                    mnist_network.x: mnist.test.images, mnist_network.y: mnist.test.labels, mnist_network.keep_prob: 1.0})
                '''@nni.report_intermediate_result(test_acc)'''

        test_acc = mnist_network.accuracy.eval(feed_dict={
            mnist_network.x: mnist.test.images, mnist_network.y: mnist.test.labels, mnist_network.keep_prob: 1.0})
        '''@nni.report_final_result(test_acc)'''


def generate_default_params():
    params = {'data_dir': '/tmp/tensorflow/mnist/input_data',
              'dropout_rate': 0.5,
              'channel_1_num': 32,
              'channel_2_num': 64,
              'conv_size': 5,
              'pool_size': 2,
              'hidden_size': 1024,
              'batch_size': 50,
              'batch_num': 200,
              'learning_rate': 1e-4}
    return params

if __name__ == '__main__':
    # run command: python mnist.py --init_file_path ./init.json

    #FLAGS, unparsed = parse_command()
    #original_params = parse_init_json(FLAGS.init_file_path, {})

    #pipe_interface.set_params_to_env()
    '''@nni.get_next_parameter()'''
    try:
        params = generate_default_params()
        logger.debug('params')
        logger.debug('params update')
        main()
    except:
        logger.exception('Got some exception in while loop in mnist.py')
        raise
