# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
import tempfile
import unittest

import numpy as np
import tensorflow as tf


####
#
# This file tests pruners on 3 models:
#   A classic CNN model built by inheriting `Model`;
#   The same CNN model built with `Sequential`;
#   A naive model with only one linear layer.
#
# The CNN models are used to test layer detecting and instrumenting.
#
# The naive model is used to test mask calculation.
# It has a single 10x10 linear layer without bias, and `reduce_sum` its result.
# To help predicting pruning result, the linear layer has fixed initial weights:
#     [ [ 0.0, 1.0, 2.0, ..., 9.0 ], [0.1, 1.1, 2.1, ..., 9.1 ], ... , [0.9, 1.0, 2.9, ..., 9.9 ] ]
#
####


# This tensor is used as input of 10x10 linear layer, the first dimension is batch size
tensor1x10 = tf.constant([[1.0] * 10])

# This tensor is used as input of CNN models
image_tensor = tf.zeros([1, 10, 10, 3])


@unittest.skipIf(tf.__version__[0] != '2', 'Skip TF 1.x setup')
class TfCompressorTestCase(unittest.TestCase):
    def test_layer_detection(self):
        # Conv and dense layers should be compressed, pool and flatten should not.
        # This also tests instrumenting functionality.
        self._test_layer_detection_on_model(CnnModel())
        self._test_layer_detection_on_model(build_sequential_model())

    def _test_layer_detection_on_model(self, model):
        pruner = pruners['level'](model)
        pruner.compress()
        layer_types = sorted(type(wrapper.layer).__name__ for wrapper in pruner.wrappers)
        assert layer_types == ['Conv2D', 'Dense', 'Dense'], layer_types

    def test_level_pruner_and_export_correctness(self):
        # prune 90% : 9.0 + 9.1 + ... + 9.9 = 94.5
        model = build_naive_model()
        pruner = pruners['level'](model)
        model = pruner.compress()

        x = model(tensor1x10)
        assert x.numpy() == 94.5

        temp_dir = Path(tempfile.gettempdir())
        pruner.export_model(temp_dir / 'model', temp_dir / 'mask')

        # because exporting will uninstrument and re-instrument the model,
        # we must test the model again
        x = model(tensor1x10)
        assert x.numpy() == 94.5

        # load and test exported model
        exported_model = tf.keras.models.load_model(temp_dir / 'model')
        x = exported_model(tensor1x10)
        assert x.numpy() == 94.5

    def test_export_not_crash(self):
        for model in [CnnModel(), build_sequential_model()]:
            pruner = pruners['level'](model)
            model = pruner.compress()
            # cannot use model.build(image_tensor.shape) here
            # it fails even without compression
            # seems TF's bug, not ours
            model(image_tensor)
            pruner.export_model(tempfile.TemporaryDirectory().name)

try:
    from tensorflow.keras import Model, Sequential
    from tensorflow.keras.layers import (Conv2D, Dense, Flatten, MaxPool2D)

    from nni.algorithms.compression.tensorflow.pruning import LevelPruner

    pruners = {
        'level': (lambda model: LevelPruner(model, [{'sparsity': 0.9, 'op_types': ['default']}])),
    }

    class CnnModel(Model):
        def __init__(self):
            super().__init__()
            self.conv = Conv2D(filters=10, kernel_size=3, activation='relu')
            self.pool = MaxPool2D(pool_size=2)
            self.flatten = Flatten()
            self.fc1 = Dense(units=10, activation='relu')
            self.fc2 = Dense(units=5, activation='softmax')

        def call(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    def build_sequential_model():
        return Sequential([
            Conv2D(filters=10, kernel_size=3, activation='relu'),
            MaxPool2D(pool_size=2),
            Flatten(),
            Dense(units=10, activation='relu'),
            Dense(units=5, activation='softmax'),
        ])

    class NaiveModel(Model):
        def __init__(self):
            super().__init__()
            self.fc = Dense(units=10, use_bias=False)

        def call(self, x):
            return tf.math.reduce_sum(self.fc(x))

except Exception:
    pass


def build_naive_model():
    model = NaiveModel()
    model.build(tensor1x10.shape)
    weight = [[(i + j * 0.1) for i in range(10)] for j in range(10)]
    model.set_weights([np.array(weight)])
    return model


if __name__ == '__main__':
    unittest.main()
