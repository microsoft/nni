from unittest import TestCase, main
from nni.compressors import tfCompressor,torchCompressor
import torch
import tensorflow as tf 

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev = 0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape = shape))

def conv2d(x_input, w_matrix):
    return tf.nn.conv2d(x_input, w_matrix, strides = [ 1, 1, 1, 1 ], padding = 'SAME')

def max_pool(x_input, pool_size):
    size = [ 1, pool_size, pool_size, 1 ]
    return tf.nn.max_pool(x_input, ksize = size, strides = size, padding = 'SAME')


class TfMnist:
    def __init__(self):
        images = tf.placeholder(tf.float32, [ None, 784 ], name = 'input_x')
        labels = tf.placeholder(tf.float32, [ None, 10 ], name = 'input_y')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.images = images
        self.labels = labels
        self.keep_prob = keep_prob

        self.train_step = None
        self.accuracy = None

        self.w1 = None
        self.b1 = None
        self.fcw1 = None
        self.cross = None
        with tf.name_scope('reshape'):
            x_image = tf.reshape(images, [ -1, 28, 28, 1 ])
        with tf.name_scope('conv1'):
            w_conv1 = weight_variable([ 5, 5, 1, 32 ])
            self.w1 = w_conv1
            b_conv1 = bias_variable([ 32 ])
            self.b1 = b_conv1
            h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
        with tf.name_scope('pool1'):
            h_pool1 = max_pool(h_conv1, 2)
        with tf.name_scope('conv2'):
            w_conv2 = weight_variable([ 5, 5, 32, 64 ])
            b_conv2 = bias_variable([ 64 ])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
        with tf.name_scope('pool2'):
            h_pool2 = max_pool(h_conv2, 2)
        with tf.name_scope('fc1'):
            w_fc1 = weight_variable([ 7 * 7 * 64, 1024 ])
            self.fcw1 = w_fc1
            b_fc1 = bias_variable([ 1024 ])
        h_pool2_flat = tf.reshape(h_pool2, [ -1, 7 * 7 * 64 ])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
        with tf.name_scope('dropout'):
            h_fc1_drop = tf.nn.dropout(h_fc1, 0.5)
        with tf.name_scope('fc2'):
            w_fc2 = weight_variable([ 1024, 10 ])
            b_fc2 = bias_variable([ 10 ])
            y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
        with tf.name_scope('loss'):
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = y_conv))
            self.cross = cross_entropy
        with tf.name_scope('adam_optimizer'):
            self.train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

class TorchMnist(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.fc1 = torch.nn.Linear(4 * 4 * 50, 500)
        self.fc2 = torch.nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim = 1)

class CompressorTestCase(TestCase):
    def test_tf_pruner(self):
        model = TfMnist()
        tfCompressor.LevelPruner(sparsity = 0.8).compress_default_graph()


    def test_tf_quantizer(self):
        model = TfMnist()
        tfCompressor.NaiveQuantizer().compress_default_graph()
    
    def test_torch_pruner(self):
        model = TorchMnist()
        torchCompressor.LevelPruner(sparsity = 0.8).compress(model)
    
    def test_torch_quantizer(self):
        model = TorchMnist()
        torchCompressor.NaiveQuantizer().compress(model)


if __name__ == '__main__':
    main()
