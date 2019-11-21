# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
NNI example trial code.

- Experiment type: Hyper-parameter Optimization
- Trial framework: Tensorflow v2.x (Keras API)
- Model: LeNet-5
- Dataset: MNIST
"""

import logging

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten, MaxPool2D)
from tensorflow.keras.optimizers import Adam

import nni

_logger = logging.getLogger('mnist_example')


class MnistModel(Model):
    """
    LeNet-5 Model with customizable hyper-parameters
    """
    def __init__(self, conv_size, hidden_size, dropout_rate):
        """
        Initialize hyper-parameters.

        Parameters
        ----------
        conv_size : int
            Kernel size of convolutional layers.
        hidden_size : int
            Dimensionality of last hidden layer.
        dropout_rate : float
            Dropout rate between two fully connected (dense) layers, to prevent co-adaptation.
        """
        super().__init__()
        self.conv1 = Conv2D(32, kernel_size=conv_size, activation='relu')
        self.pool1 = MaxPool2D(pool_size=2)
        self.conv2 = Conv2D(filters=64, kernel_size=conv_size, activation='relu')
        self.pool2 = MaxPool2D(pool_size=2)
        self.flatten = Flatten()
        self.fc1 = Dense(units=hidden_size, activation='relu')
        self.dropout = Dropout(rate=dropout_rate)
        self.fc2 = Dense(units=10, activation='softmax')

    def call(self, x):
        """Override ``Model.call`` to build LeNet-5 model."""
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        return self.fc2(x)


class ReportIntermediates(Callback):
    """
    Callback class for reporting intermediate accuracy metrics.

    This callback sends accuracy to NNI framework every 100 steps,
    so you can view the learning curve on web UI.

    If an assessor is configured in experiment's YAML file,
    it will use these metrics for early stopping.
    """

    def on_batch_end(self, batch, logs=None):
        """Reports accuracy for every 100 steps"""
        if logs and logs.get('batch', 1) % 100 == 0:
            _logger.info('(Step %s) Intermediate accuracy: %s', logs['batch'], logs['accuracy'])
            nni.report_intermediate_result(logs['accuracy'])  # send accuracy to NNI framework


def load_dataset():
    """Download and reformat MNIST dataset"""
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    return (x_train, y_train), (x_test, y_test)


def main(params):
    """
    Main program:
      - Build network
      - Prepare dataset
      - Train the model
      - Report accuracy to tuner
    """
    model = MnistModel(
        conv_size=params['conv_size'],
        hidden_size=params['hidden_size'],
        dropout_rate=params['dropout_rate']
    )
    optimizer = Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    (x_train, y_train), (x_test, y_test) = load_dataset()

    model.fit(
        x_train,
        y_train,
        batch_size=params['batch_size'],
        epochs=1,
        verbose=0,
        callbacks=[ReportIntermediates()]
    )

    result = model.evaluate(x_test, y_test)
    _logger.info('Final accuracy: %s', result[1])
    nni.report_final_result(result[1])  # send final accuracy to NNI tuner and web UI


if __name__ == '__main__':
    default_params = {
        'dropout_rate': 0.5,
        'conv_size': 5,
        'hidden_size': 1024,
        'batch_size': 32,
        'learning_rate': 1e-4,
    }

    # fetch hyper-parameters from HPO tuner
    # comment out following two lines to run the code without NNI framework
    tuned_params = nni.get_next_parameter()
    params = {**default_params, **tuned_params}

    _logger.info('Hyper-parameters: %s', params)
    main(params)
