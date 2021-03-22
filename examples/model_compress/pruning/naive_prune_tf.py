# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

'''
NNI example for quick start of pruning.
In this example, we use level pruner to prune the LeNet on MNIST.
'''

import argparse

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten, MaxPool2D)

from nni.algorithms.compression.tensorflow.pruning import LevelPruner

class LeNet(Model):
    """
    LeNet-5 Model with customizable hyper-parameters
    """
    def __init__(self, conv_size=3, hidden_size=32, dropout_rate=0.5):
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
        self.conv1 = Conv2D(filters=32, kernel_size=conv_size, activation='relu')
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


def get_dataset(dataset_name='mnist'):
    assert dataset_name == 'mnist'

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train[..., tf.newaxis] / 255.0
    x_test = x_test[..., tf.newaxis] / 255.0
    return (x_train, y_train), (x_test, y_test)


# def create_model(model_name='naive'):
#     assert model_name == 'naive'
#     return tf.keras.Sequential([
#         tf.keras.layers.Conv2D(filters=20, kernel_size=5),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.ReLU(),
#         tf.keras.layers.MaxPool2D(pool_size=2),
#         tf.keras.layers.Conv2D(filters=20, kernel_size=5),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.ReLU(),
#         tf.keras.layers.MaxPool2D(pool_size=2),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(units=500),
#         tf.keras.layers.ReLU(),
#         tf.keras.layers.Dense(units=10),
#         tf.keras.layers.Softmax()
#     ])

def main(args):
    train_set, test_set = get_dataset('mnist')
    model = LeNet()

    print('start training')
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, decay=1e-4)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        train_set[0],
        train_set[1],
        batch_size=args.batch_size,
        epochs=args.pretrain_epochs,
        validation_data=test_set
    )

    print('start pruning')
    optimizer_finetune = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, decay=1e-4)

    # create_pruner
    prune_config = [{
        'sparsity': args.sparsity,
        'op_types': ['default'],
    }]
    
    pruner = LevelPruner(model, prune_config)
    # pruner = create_pruner(model, args.pruner_name)
    model = pruner.compress()

    model.compile(
        optimizer=optimizer_finetune,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        run_eagerly=True  # NOTE: Important, model compression does not work in graph mode!
    )

    # fine-tuning
    model.fit(
        train_set[0],
        train_set[1],
        batch_size=args.batch_size,
        epochs=args.prune_epochs,
        validation_data=test_set
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pruner_name', type=str, default='level')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--pretrain_epochs', type=int, default=10)
    parser.add_argument('--prune_epochs', type=int, default=10)
    parser.add_argument('--sparsity', type=float, default=0.5)

    args = parser.parse_args()
    main(args)
