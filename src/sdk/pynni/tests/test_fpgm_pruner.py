from unittest import TestCase, main
import numpy as np
import tensorflow as tf
import torch
import nni.compression.torch as torch_compressor

if tf.__version__ >= '2.0':
    import nni.compression.tensorflow as tf_compressor

def get_tf_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=10, kernel_size=3, input_shape=[28, 28, 5], activation='relu', padding="SAME"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=10, activation='softmax'),
    ])
    model.compile(loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.SGD(lr=1e-3),
        metrics=["accuracy"])
    return model

class TorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(5, 10, 3)

    def forward(self, x):
        return self.conv1(x)

def tf2(func):
    def test_tf2_func(*args):
        if tf.__version__ >= '2.0':
            func(*args)
    return test_tf2_func

'''
With filters(kernels) defined as below (k1 - k5), it is obvious that k3 is the Geometric Median
which minimize the total geometric distance by defination of Geometric Median in this paper:
Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration,
https://arxiv.org/pdf/1811.00250.pdf

So if sparsity is 0.2, the expected masks should mask out all k3, this can be verified through:
`all(torch.sum(masks, (0, 2, 3)).numpy() == np.array([90., 90., 0., 90., 90.]))`

If sparsity is 0.6, the expected masks should mask out all k2, k3, k4, this can be verified through:
`all(torch.sum(masks, (0, 2, 3)).numpy() == np.array([90., 0., 0., 0., 90.]))`
'''

k1 = [[1]*3]*3
k2 = [[2]*3]*3
k3 = [[3]*3]*3
k4 = [[4]*3]*3
k5 = [[5]*3]*3

w = [[k1, k2, k3, k4, k5]] * 10


class FPGMTestCase(TestCase):
    def test_torch_fpgm_pruner(self):
        model = TorchModel()
        config_list = [{'sparsity': 0.2, 'op_types': ['Conv2d']}, {'sparsity': 0.6, 'op_types': ['Conv2d']}]
        pruner = torch_compressor.FPGMPruner(model, config_list)

        model.conv1.weight.data = torch.tensor(w).float()
        layer = torch_compressor.compressor.LayerInfo('conv1', model.conv1)
        masks = pruner.calc_mask(layer, config_list[0])
        assert all(torch.sum(masks, (0, 2, 3)).numpy() == np.array([90., 90., 0., 90., 90.]))

        pruner.update_epoch(1)
        model.conv1.weight.data = torch.tensor(w).float()
        masks = pruner.calc_mask(layer, config_list[1])
        assert all(torch.sum(masks, (0, 2, 3)).numpy() == np.array([90., 0., 0., 0., 90.]))

    @tf2
    def test_tf_fpgm_pruner(self):
        model = get_tf_model()
        config_list = [{'sparsity': 0.2, 'op_types': ['Conv2D']}, {'sparsity': 0.6, 'op_types': ['Conv2D']}]

        pruner = tf_compressor.FPGMPruner(model, config_list)
        weights = model.layers[0].weights
        weights[0] = np.array(w).astype(np.float32).transpose([2, 3, 0, 1]).transpose([0, 1, 3, 2])
        model.layers[0].set_weights([weights[0], weights[1].numpy()])

        layer = tf_compressor.compressor.LayerInfo(model.layers[0])
        masks = pruner.calc_mask(layer, config_list[0]).numpy()
        masks = masks.transpose([2, 3, 0, 1]).transpose([1, 0, 2, 3])

        assert all(masks.sum((0, 2, 3)) == np.array([90., 90., 0., 90., 90.]))

        pruner.update_epoch(1)
        model.layers[0].set_weights([weights[0], weights[1].numpy()])
        masks = pruner.calc_mask(layer, config_list[1]).numpy()
        masks = masks.transpose([2, 3, 0, 1]).transpose([1, 0, 2, 3])

        assert all(masks.sum((0, 2, 3)) == np.array([90., 0., 0., 0., 90.]))

if __name__ == '__main__':
    main()
