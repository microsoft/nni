from nni.compressors.tf_compressor import AGPruner
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev = 0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape = shape))

def conv2d(x_input, w_matrix):
    return tf.nn.conv2d(x_input, w_matrix, strides = [ 1, 1, 1, 1 ], padding = 'SAME')

def max_pool(x_input, pool_size):
    size = [ 1, pool_size, pool_size, 1 ]
    return tf.nn.max_pool(x_input, ksize = size, strides = size, padding = 'SAME')


class Mnist:
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


def main():
    tf.set_random_seed(0)

    data = input_data.read_data_sets('data', one_hot = True)

    model = Mnist()

    '''you can change this to SensitivityPruner to implement it
    pruner = SensitivityPruner(configure_list)
    '''
    configure_list = [{
                        'initial_sparsity': 0,
                        'final_sparsity': 0.8,
                        'start_epoch': 1,
                        'end_epoch': 10,
                        'frequency': 1,
                        'support_type': 'default'
                    }]
    pruner = AGPruner(configure_list)
    pruner(tf.get_default_graph())
    # you can also use compress(model) or compress_default_graph() for tensorflow compressor
    # pruner.compress(tf.get_default_graph())
    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for batch_idx in range(2000):
            batch = data.train.next_batch(2000)
            model.train_step.run(feed_dict = {
                model.images: batch[0],
                model.labels: batch[1],
                model.keep_prob: 0.5
            })
            if batch_idx % 10 == 0:
                test_acc = model.accuracy.eval(feed_dict = {
                    model.images: data.test.images,
                    model.labels: data.test.labels,
                    model.keep_prob: 1.0
                })
                pruner.update_epoch(batch_idx / 10,sess)
                print('test accuracy', test_acc)
                
        
        test_acc = model.accuracy.eval(feed_dict = {
            model.images: data.test.images,
            model.labels: data.test.labels,
            model.keep_prob: 1.0
        })
        print('final result is', test_acc)

main()
