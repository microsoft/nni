# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import logging
import os
import sys
import time
import math

import utils
import nni
import numpy as np
import tensorflow as tf
from keras.datasets import cifar10

import onnx
import mmdnn
import warnings
from onnx_tf.backend import prepare, TensorflowRep
from onnx_tf.common import get_output_node_names
from onnx_tf.frontend import tensorflow_graph_to_onnx_model

# set the logger format
logger = logging.getLogger('cifar10-network-morphism')
log_format = '%(asctime)s %(message)s'
logger.basicConfig(stream=sys.stdout, level=logging.INFO,
                   format=log_format, datefmt='%m/%d %I:%M:%S %p')

optimizer = None
train_init_op = None
test_init_op = None
train_features = None
train_labels = None
test_features = None
test_labels = None
train_step = 0
test_step = 0
best_acc = 0.0
args = get_args()


def get_args():
    parser = argparse.ArgumentParser("cifar10")
    parser.add_argument('--batch_size', type=int,
                        default=96, help='batch size')
    parser.add_argument('--optimizer', type=str,
                        default="Adam", help='optimizer')
    parser.add_argument('--epoches', type=int, default=30, help='epoch limit')
    parser.add_argument('--learning_rate', type=float,
                        default=1e-3, help='epoch limit')
    parser.add_argument('--time_limit', type=int,
                        default=0, help='gpu device id')
    parser.add_argument('--cutout', action='store_true',
                        default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int,
                        default=16, help='cutout length')
    parser.add_argument('--model_path', type=str, default="./",
                        help='Path to save the destination model')
    args = parser.parse_args()
    return args


def build_graph_from_onnx(onnx_model_path):
    ''' build model from onnx intermedia represtation 
    '''

    # load onnx model from model path
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    onnx.helper.printable_graph(onnx_model)

    # Polishing the Model
    # polished_model = onnx.utils.polish_model(onnx_model)

    # Ignore all the warning messages in this tutorial
    warnings.filterwarnings('ignore')

    # Import the ONNX model to Tensorflow
    tf_rep = prepare(onnx_model)
    # tf_rep.export_graph(args.model_path)
    graph_def = tf_rep.graph.as_graph_def()
    return graph_def


def save_graph_to_onnx(onnx_model_path):
    ''' save model to onnx intermedia represtation 
    '''

    # load tf graph def
    # graph_def = tf.GraphDef()
    # with open(args.model_path, "rb") as f:
    #     graph_def.ParseFromString(f.read())

    graph_def = tf.get_default_graph().as_graph_def()

    # get output node names
    output = get_output_node_names(graph_def)

    # convert tf graph to onnx model
    model = tensorflow_graph_to_onnx_model(graph_def, output)
    with open(onnx_model_path, 'wb') as f:
        f.write(model.SerializeToString())


def parse_rev_args(receive_msg):
    global train_init_op
    global test_init_op
    global train_features
    global train_labels
    global test_features
    global test_labels
    global train_step
    global test_step
    global optimizer

    # Data
    logger.info('Preparing data..')
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    (trainX, trainY) = (x_train.astype(np.uint8), y_train.astype(np.int32))
    (testX, testY) = (x_test.astype(np.uint8), y_test.astype(np.int32))
    train_step = np.ceil(len(trainX) / args.bacch_size)
    test_step = np.ceil(len(testX) / args.bacch_size)
    train_dataset = tf.data.Dataset.from_tensor_slices((trainX, trainY))
    test_dataset = tf.data.Dataset.from_tensor_slices((testX, testY))
    train_dataset = train_dataset.repeat().shuffle(10000)
    train_dataset = train_dataset.map(map_single, num_parallel_calls=8)
    train_dataset = train_dataset.batch(args.batch_size)
    train_dataset = train_dataset.map(map_batch)
    train_dataset = train_dataset.prefetch(2)

    test_dataset = test_dataset.map(map_single, num_parallel_calls=8).batch(
        args.batch_size).map(map_batch)
    test_dataset = test_dataset.prefetch(2)

    train_iterator = tf.data.Iterator.from_structure(
        train_dataset.output_types, train_dataset.output_shapes)
    test_iterator = tf.data.Iterator.from_structure(
        test_dataset.output_types, test_dataset.output_shapes)
    train_features, train_labels = train_iterator.get_next()
    test_features, test_labels = test_iterator.get_next()
    train_init_op = train_iterator.make_initializer(train_dataset)
    test_init_op = test_iterator.make_initializer(test_dataset)
    logger.info('Preparing successfully.')

    # Model
    logger.info('Building model..')
    model_path = receive_msg
    graph_def = build_graph_from_onnx(model_path)

    if args.optimizer == 'SGD':
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=args.learning_rate)
    elif args.optimizer == 'Adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate=args.learning_rate)
    elif args.optimizer == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=args.learning_rate)
    elif args.optimizer == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    elif args.optimizer == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=args.learning_rate, momentum=0.9)
    else:
        raise RuntimeError("{} optimizer not supported".format(args.optimizer))

    return graph_def

# Training


def train(sess, epoch):
    global train_init_op
    global train_features
    global train_labels
    global train_step
    global optimizer

    logger.info('Epoch: %d' % epoch)
    # Placeholders
    x = tf.get_default_graph().get_tensor_by_name('x:0')
    y = tf.get_default_graph().get_tensor_by_name('y:0')
    is_training = tf.get_default_graph().get_tensor_by_name('is_training:0')
    loss = tf.get_default_graph().get_tensor_by_name('loss:0')
    out = tf.get_default_graph().get_tensor_by_name('output:0')
    acc = tf.get_default_graph().get_tensor_by_name('accuary:0')
    
    train_loss = 0
    total = 0
    sess.run(train_init_op)
    for _ in range(train_step):
        batch_x, batch_y = sess.run((train_features, train_labels))
        train_feed_dict = {
            x: batch_x,
            y: batch_y,
            is_training : True
        }
        _ , batch_loss = sess.run([optimizer,loss], feed_dict=train_feed_dict)
        train_loss += batch_loss
        total+=batch_x.shape(0)
         



def test(sess, epoch, onnx_model_path):
    global best_acc
    global test_init_op
    global test_features
    global test_labels
    global test_step
    global optimizer

    # Placeholders
    x = tf.get_default_graph().get_tensor_by_name('x:0')
    y = tf.get_default_graph().get_tensor_by_name('y:0')
    is_training = tf.get_default_graph().get_tensor_by_name('is_training:0')
    loss = tf.get_default_graph().get_tensor_by_name('loss:0')
    out = tf.get_default_graph().get_tensor_by_name('output:0')
    acc = tf.get_default_graph().get_tensor_by_name('accuary:0')
    feed = dict()

    test_loss = 0
    correct = 0
    total = 0
    sess.run(test_init_op)
    for _ in range(train_step):
        batch_x, batch_y = sess.run((test_features, test_labels))
        test_feed_dict = {
            x: batch_x,
            y: batch_y,
            is_training : False
        }
        _ ,acc_batch, batch_loss = sess.run([optimizer,acc,loss], feed_dict=test_feed_dict)
        test_loss += batch_loss
        total+=batch_x.shape(0)
        correct+=acc_batch
   
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        logger.info('Saving..')
        save_graph_to_onnx(onnx_model_path)
        best_acc = acc
    return acc, best_acc


def map_single(x, y):
    print('Map single:')
    print('x shape: %s' % str(x.shape))
    print('y shape: %s' % str(y.shape))
    x = tf.image.per_image_standardization(x)
    # Consider: x = tf.image.random_flip_left_right(x)
    return x, y


def map_batch(x, y):
    print('Map batch:')
    print('x shape: %s' % str(x.shape))
    print('y shape: %s' % str(y.shape))
    # Note: this flips ALL images left to right. Not sure this is what you want
    # UPDATE: looks like tf documentation is wrong and you need a 3D tensor?
    # return tf.image.flip_left_right(x), y
    return x, y


def augment(images, labels, resize=None, horizontal_flip=False, vertical_flip=False,
            rotate=0, crop_probability=0, crop_min_percent=0.6, crop_max_percent=1., mixup=0):
    ''' data augment using tensorflow

    Arguments:
        images,labels
    Keyword Arguments:
        resize {tuple} --(width, height) tuple or None (default: {None})
        horizontal_flip {bool} -- [description] (default: {False})
        vertical_flip {bool} -- [description] (default: {False})
        rotate {int} -- Maximum rotation angle in degrees (default: {0})
        crop_probability {int} -- How often we do crops (default: {0})
        crop_min_percent {float} -- Minimum linear dimension of a crop (default: {0.6})
        crop_max_percent {float} -- Maximum linear dimension of a crop
        mixup {int} -- Mixup coeffecient, see https://arxiv.org/abs/1710.09412.pdf

    Returns:
       images, labels 
    '''

    if resize is not None:
        images = tf.image.resize_bilinear(images, resize)

    # casting on GPU improves training performance
    if images.dtype != tf.float32:
        images = tf.image.convert_image_dtype(images, dtype=tf.float32)
        images = tf.subtract(images, 0.5)
        images = tf.multiply(images, 2.0)
    labels = tf.to_float(labels)

    with tf.name_scope('augmentation'):
        shp = tf.shape(images)
        batch_size, height, width = shp[0], shp[1], shp[2]
        width = tf.cast(width, tf.float32)
        height = tf.cast(height, tf.float32)

        # The list of affine transformations that our image will go under.
        # Every element is Nx8 tensor, where N is a batch size.
        transforms = []
        identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
        if horizontal_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            flip_transform = tf.convert_to_tensor(
                [-1., 0., width, 0., 1., 0., 0., 0.], dtype=tf.float32)
            transforms.append(
                tf.where(coin,
                         tf.tile(tf.expand_dims(flip_transform, 0),
                                 [batch_size, 1]),
                         tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if vertical_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            flip_transform = tf.convert_to_tensor(
                [1, 0, 0, 0, -1, height, 0, 0], dtype=tf.float32)
            transforms.append(
                tf.where(coin,
                         tf.tile(tf.expand_dims(flip_transform, 0),
                                 [batch_size, 1]),
                         tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if rotate > 0:
            angle_rad = rotate / 180 * math.pi
            angles = tf.random_uniform([batch_size], -angle_rad, angle_rad)
            transforms.append(
                tf.contrib.image.angles_to_projective_transforms(
                    angles, height, width))

        if crop_probability > 0:
            crop_pct = tf.random_uniform([batch_size], crop_min_percent,
                                         crop_max_percent)
            left = tf.random_uniform([batch_size], 0, width * (1 - crop_pct))
            top = tf.random_uniform([batch_size], 0, height * (1 - crop_pct))
            crop_transform = tf.stack([
                crop_pct,
                tf.zeros([batch_size]), top,
                tf.zeros([batch_size]), crop_pct, left,
                tf.zeros([batch_size]),
                tf.zeros([batch_size])
            ], 1)

            coin = tf.less(
                tf.random_uniform([batch_size], 0, 1.0), crop_probability)
            transforms.append(
                tf.where(coin, crop_transform,
                         tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if transforms:
            images = tf.contrib.image.transform(
                images,
                tf.contrib.image.compose_transforms(*transforms),
                interpolation='BILINEAR')  # or 'NEAREST'

        def cshift(values):  # Circular shift in batch dimension
            return tf.concat([values[-1:, ...], values[:-1, ...]], 0)

        if mixup > 0:
            # Convert to float, as tf.distributions.Beta requires floats.
            mixup = 1.0 * mixup
            beta = tf.distributions.Beta(mixup, mixup)
            lam = beta.sample(batch_size)
            ll = tf.expand_dims(tf.expand_dims(
                tf.expand_dims(lam, -1), -1), -1)
            images = ll * images + (1 - ll) * cshift(images)
            labels = lam * labels + (1 - lam) * cshift(labels)

    return images, labels


def main():
    try:
        # trial get next parameter from network morphism tuner
        RCV_CONFIG = nni.get_next_parameter()
        logger.debug(RCV_CONFIG)

        graph_def = parse_rev_args(RCV_CONFIG)

        acc = 0.0
        best_acc = 0.0
        with tf.Session() as sess:
            # import graph from graph_def
            tf.import_graph_def(graph_def, name="default graph define")
            for epoch in range(args.epoches):
                train(sess, epoch)
                acc, best_acc = test(sess, epoch, RCV_CONFIG)
                nni.report_intermediate_result(acc)

        # trial report best_acc to tuner
        nni.report_final_result(best_acc)
    except Exception as exception:
        logger.exception(exception)
        raise


if __name__ == '__main__':
    main()
