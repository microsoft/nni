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

import keras
import keras.backend.tensorflow_backend as KTF
import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, Nadam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model, to_categorical
from keras.callbacks import TensorBoard, EarlyStopping
import nni
from nni.networkmorphism_tuner.graph import json_to_graph

# set the logger format
log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    filename="networkmorphism.log",
    filemode="a",
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)
# set the logger format
logger = logging.getLogger("cifar10-network-morphism-keras")


# restrict gpu usage background
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

KTF.set_session(sess)


def get_args():
    parser = argparse.ArgumentParser("cifar10")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--optimizer", type=str, default="SGD", help="optimizer")
    parser.add_argument("--epochs", type=int, default=200, help="epoch limit")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="weight decay of the learning rate",
    )
    args = parser.parse_args()
    return args


trainloader = None
testloader = None
net = None
args = get_args()
TENSORBOARD_DIR = os.environ["NNI_OUTPUT_DIR"]


def build_graph_from_json(ir_model_json):
    """build model from json representation
    """
    graph = json_to_graph(ir_model_json)
    logging.debug(graph.operation_history)
    model = graph.produce_keras_model()
    return model


def parse_rev_args(receive_msg):
    global trainloader
    global testloader
    global datagen
    global net

    # Loading Data
    logger.debug("Preparing data..")

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255.0
    x_test /= 255.0
    trainloader = (x_train, y_train)
    testloader = (x_test, y_test)

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
    )
    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(x_train)

    # Model
    logger.debug("Building model..")
    net = build_graph_from_json(receive_msg)

    # parallel model
    try:
        available_devices = os.environ["CUDA_VISIBLE_DEVICES"]
        gpus = len(available_devices.split(","))
        if gpus > 1:
            net = multi_gpu_model(net, gpus)
    except KeyError:
        logger.debug("parallel model not support in this config settings")

    if args.optimizer == "SGD":
        optimizer = SGD(lr=args.learning_rate, momentum=0.9, decay=args.weight_decay)
    if args.optimizer == "Adadelta":
        optimizer = Adadelta(lr=args.learning_rate, decay=args.weight_decay)
    if args.optimizer == "Adagrad":
        optimizer = Adagrad(lr=args.learning_rate, decay=args.weight_decay)
    if args.optimizer == "Adam":
        optimizer = Adam(lr=args.learning_rate, decay=args.weight_decay)
    if args.optimizer == "Adamax":
        optimizer = Adamax(lr=args.learning_rate, decay=args.weight_decay)

    # Compile the model
    net.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    return 0


class SendMetrics(keras.callbacks.Callback):
    """
    Keras callback to send metrics to NNI framework
    """

    def on_epoch_end(self, epoch, logs={}):
        """
        Run on end of each epoch
        """
        logger.debug(logs)
        nni.report_intermediate_result(logs["acc"])


# Training
def train_eval():
    global trainloader
    global testloader
    global datagen
    global net

    (x_train, y_train) = trainloader
    (x_test, y_test) = testloader

    # train procedure
    net.fit(
        x=x_train,
        y=y_train,
        batch_size=args.batch_size,
        validation_data=(x_test, y_test),
        epochs=args.epochs,
        shuffle=True,
        callbacks=[
            SendMetrics(),
            EarlyStopping(min_delta=0.001, patience=10),
            TensorBoard(log_dir=TENSORBOARD_DIR),
        ],
    )

    # trial report final acc to tuner
    _, acc = net.evaluate(x_test, y_test)
    logger.debug("Final result is: %d", acc)
    nni.report_final_result(acc)


if __name__ == "__main__":
    try:
        # trial get next parameter from network morphism tuner
        RCV_CONFIG = nni.get_next_parameter()
        logger.debug(RCV_CONFIG)
        parse_rev_args(RCV_CONFIG)
        train_eval()
    except Exception as exception:
        logger.exception(exception)
        raise
