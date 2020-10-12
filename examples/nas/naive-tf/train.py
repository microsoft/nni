import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (AveragePooling2D, BatchNormalization, Conv2D, Dense, MaxPool2D)
from tensorflow.keras.losses import Reduction, SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD

from nni.nas.tensorflow.mutables import LayerChoice, InputChoice
from nni.nas.tensorflow.enas import EnasTrainer

tf.get_logger().setLevel('ERROR')


class Net(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = LayerChoice([
            Conv2D(6, 3, padding='same', activation='relu'),
            Conv2D(6, 5, padding='same', activation='relu'),
        ])
        self.pool = MaxPool2D(2)
        self.conv2 = LayerChoice([
            Conv2D(16, 3, padding='same', activation='relu'),
            Conv2D(16, 5, padding='same', activation='relu'),
        ])
        self.conv3 = Conv2D(16, 1)

        self.skipconnect = InputChoice(n_candidates=1)
        self.bn = BatchNormalization()

        self.gap = AveragePooling2D(2)
        self.fc1 = Dense(120, activation='relu')
        self.fc2 = Dense(84, activation='relu')
        self.fc3 = Dense(10)

    def call(self, x):
        bs = x.shape[0]

        t = self.conv1(x)
        x = self.pool(t)
        x0 = self.conv2(x)
        x1 = self.conv3(x0)

        x0 = self.skipconnect([x0])
        if x0 is not None:
            x1 += x0
        x = self.pool(self.bn(x1))

        x = self.gap(x)
        x = tf.reshape(x, [bs, -1])
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def accuracy(output, target):
    bs = target.shape[0]
    predicted = tf.cast(tf.argmax(output, 1), target.dtype)
    target = tf.reshape(target, [-1])
    return sum(tf.cast(predicted == target, tf.float32)) / bs


if __name__ == '__main__':
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_valid, y_valid) = cifar10.load_data()
    x_train, x_valid = x_train / 255.0, x_valid / 255.0
    train_set = (x_train, y_train)
    valid_set = (x_valid, y_valid)

    net = Net()

    trainer = EnasTrainer(
        net,
        loss=SparseCategoricalCrossentropy(reduction=Reduction.SUM),
        metrics=accuracy,
        reward_function=accuracy,
        optimizer=SGD(learning_rate=0.001, momentum=0.9),
        batch_size=64,
        num_epochs=2,
        dataset_train=train_set,
        dataset_valid=valid_set
    )

    trainer.train()
