# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest import TestCase, main
import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
import nni.compression.torch as torch_compressor
import math

if tf.__version__ >= '2.0':
    import nni.compression.tensorflow as tf_compressor


def get_tf_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=5, kernel_size=7, input_shape=[28, 28, 1], activation='relu', padding="SAME"),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Conv2D(filters=10, kernel_size=3, activation='relu', padding="SAME"),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(units=10, activation='softmax'),
    ])
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=tf.keras.optimizers.SGD(lr=1e-3),
                  metrics=["accuracy"])
    return model


class TorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 5, 5, 1)
        self.bn1 = torch.nn.BatchNorm2d(5)
        self.conv2 = torch.nn.Conv2d(5, 10, 5, 1)
        self.bn2 = torch.nn.BatchNorm2d(10)
        self.fc1 = torch.nn.Linear(4 * 4 * 10, 100)
        self.fc2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 10)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def tf2(func):
    def test_tf2_func(*args):
        if tf.__version__ >= '2.0':
            func(*args)

    return test_tf2_func


# for fpgm filter pruner test
w = np.array([[[[i + 1] * 3] * 3] * 5 for i in range(10)])


class CompressorTestCase(TestCase):
    def test_torch_quantizer_modules_detection(self):
        # test if modules can be detected
        model = TorchModel()
        config_list = [{
            'quant_types': ['weight'],
            'quant_bits': 8,
            'op_types': ['Conv2d', 'Linear']
        }, {
            'quant_types': ['output'],
            'quant_bits': 8,
            'quant_start_step': 0,
            'op_types': ['ReLU']
        }]

        model.relu = torch.nn.ReLU()
        quantizer = torch_compressor.QAT_Quantizer(model, config_list)
        quantizer.compress()
        modules_to_compress = quantizer.get_modules_to_compress()
        modules_to_compress_name = [t[0].name for t in modules_to_compress]
        assert "conv1" in modules_to_compress_name
        assert "conv2" in modules_to_compress_name
        assert "fc1" in modules_to_compress_name
        assert "fc2" in modules_to_compress_name
        assert "relu" in modules_to_compress_name
        assert len(modules_to_compress_name) == 5

    def test_torch_level_pruner(self):
        model = TorchModel()
        configure_list = [{'sparsity': 0.8, 'op_types': ['default']}]
        torch_compressor.LevelPruner(model, configure_list).compress()

    @tf2
    def test_tf_level_pruner(self):
        configure_list = [{'sparsity': 0.8, 'op_types': ['default']}]
        tf_compressor.LevelPruner(get_tf_model(), configure_list).compress()

    def test_torch_naive_quantizer(self):
        model = TorchModel()
        configure_list = [{
            'quant_types': ['weight'],
            'quant_bits': {
                'weight': 8,
            },
            'op_types': ['Conv2d', 'Linear']
        }]
        torch_compressor.NaiveQuantizer(model, configure_list).compress()

    @tf2
    def test_tf_naive_quantizer(self):
        tf_compressor.NaiveQuantizer(get_tf_model(), [{'op_types': ['default']}]).compress()

    def test_torch_fpgm_pruner(self):
        """
        With filters(kernels) weights defined as above (w), it is obvious that w[4] and w[5] is the Geometric Median
        which minimize the total geometric distance by defination of Geometric Median in this paper:
        Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration,
        https://arxiv.org/pdf/1811.00250.pdf

        So if sparsity is 0.2, the expected masks should mask out w[4] and w[5], this can be verified through:
        `all(torch.sum(masks, (1, 2, 3)).numpy() == np.array([45., 45., 45., 45., 0., 0., 45., 45., 45., 45.]))`

        If sparsity is 0.6, the expected masks should mask out w[2] - w[7], this can be verified through:
        `all(torch.sum(masks, (1, 2, 3)).numpy() == np.array([45., 45., 0., 0., 0., 0., 0., 0., 45., 45.]))`
        """

        model = TorchModel()
        config_list = [{'sparsity': 0.2, 'op_types': ['Conv2d']}, {'sparsity': 0.6, 'op_types': ['Conv2d']}]
        pruner = torch_compressor.FPGMPruner(model, config_list)

        model.conv2.weight.data = torch.tensor(w).float()
        layer = torch_compressor.compressor.LayerInfo('conv2', model.conv2)
        masks = pruner.calc_mask(layer, config_list[0])
        assert all(torch.sum(masks, (1, 2, 3)).numpy() == np.array([45., 45., 45., 45., 0., 0., 45., 45., 45., 45.]))

        pruner.update_epoch(1)
        model.conv2.weight.data = torch.tensor(w).float()
        masks = pruner.calc_mask(layer, config_list[1])
        assert all(torch.sum(masks, (1, 2, 3)).numpy() == np.array([45., 45., 0., 0., 0., 0., 0., 0., 45., 45.]))

    @tf2
    def test_tf_fpgm_pruner(self):
        model = get_tf_model()
        config_list = [{'sparsity': 0.2, 'op_types': ['Conv2D']}, {'sparsity': 0.6, 'op_types': ['Conv2D']}]

        pruner = tf_compressor.FPGMPruner(model, config_list)
        weights = model.layers[2].weights
        weights[0] = np.array(w).astype(np.float32).transpose([2, 3, 0, 1]).transpose([0, 1, 3, 2])
        model.layers[2].set_weights([weights[0], weights[1].numpy()])

        layer = tf_compressor.compressor.LayerInfo(model.layers[2])
        masks = pruner.calc_mask(layer, config_list[0]).numpy()
        masks = masks.reshape((-1, masks.shape[-1])).transpose([1, 0])

        assert all(masks.sum((1)) == np.array([45., 45., 45., 45., 0., 0., 45., 45., 45., 45.]))

        pruner.update_epoch(1)
        model.layers[2].set_weights([weights[0], weights[1].numpy()])
        masks = pruner.calc_mask(layer, config_list[1]).numpy()
        masks = masks.reshape((-1, masks.shape[-1])).transpose([1, 0])
        assert all(masks.sum((1)) == np.array([45., 45., 0., 0., 0., 0., 0., 0., 45., 45.]))

    def test_torch_l1filter_pruner(self):
        """
        Filters with the minimum sum of the weights' L1 norm are pruned in this paper:
        PRUNING FILTERS FOR EFFICIENT CONVNETS,
        https://arxiv.org/abs/1608.08710

        So if sparsity is 0.2, the expected masks should mask out filter 0, this can be verified through:
        `all(torch.sum(mask1, (1, 2, 3)).numpy() == np.array([0., 27., 27., 27., 27.]))`

        If sparsity is 0.6, the expected masks should mask out filter 0,1,2, this can be verified through:
        `all(torch.sum(mask2, (1, 2, 3)).numpy() == np.array([0., 0., 0., 27., 27.]))`
        """
        w = np.array([np.zeros((3, 3, 3)), np.ones((3, 3, 3)), np.ones((3, 3, 3)) * 2,
                      np.ones((3, 3, 3)) * 3, np.ones((3, 3, 3)) * 4])
        model = TorchModel()
        config_list = [{'sparsity': 0.2, 'op_types': ['Conv2d'], 'op_names': ['conv1']},
                       {'sparsity': 0.6, 'op_types': ['Conv2d'], 'op_names': ['conv2']}]
        pruner = torch_compressor.L1FilterPruner(model, config_list)

        model.conv1.weight.data = torch.tensor(w).float()
        model.conv2.weight.data = torch.tensor(w).float()
        layer1 = torch_compressor.compressor.LayerInfo('conv1', model.conv1)
        mask1 = pruner.calc_mask(layer1, config_list[0])
        layer2 = torch_compressor.compressor.LayerInfo('conv2', model.conv2)
        mask2 = pruner.calc_mask(layer2, config_list[1])
        assert all(torch.sum(mask1, (1, 2, 3)).numpy() == np.array([0., 27., 27., 27., 27.]))
        assert all(torch.sum(mask2, (1, 2, 3)).numpy() == np.array([0., 0., 0., 27., 27.]))

    def test_torch_slim_pruner(self):
        """
        Scale factors with minimum l1 norm in the BN layers are pruned in this paper:
        Learning Efficient Convolutional Networks through Network Slimming,
        https://arxiv.org/pdf/1708.06519.pdf

        So if sparsity is 0.2, the expected masks should mask out channel 0, this can be verified through:
        `all(mask1.numpy() == np.array([0., 1., 1., 1., 1.]))`
        `all(mask2.numpy() == np.array([0., 1., 1., 1., 1.]))`

        If sparsity is 0.6, the expected masks should mask out channel 0,1,2, this can be verified through:
        `all(mask1.numpy() == np.array([0., 0., 0., 1., 1.]))`
        `all(mask2.numpy() == np.array([0., 0., 0., 1., 1.]))`
        """
        w = np.array([0, 1, 2, 3, 4])
        model = TorchModel()
        config_list = [{'sparsity': 0.2, 'op_types': ['BatchNorm2d']}]
        model.bn1.weight.data = torch.tensor(w).float()
        model.bn2.weight.data = torch.tensor(-w).float()
        pruner = torch_compressor.SlimPruner(model, config_list)

        layer1 = torch_compressor.compressor.LayerInfo('bn1', model.bn1)
        mask1 = pruner.calc_mask(layer1, config_list[0])
        layer2 = torch_compressor.compressor.LayerInfo('bn2', model.bn2)
        mask2 = pruner.calc_mask(layer2, config_list[0])
        assert all(mask1.numpy() == np.array([0., 1., 1., 1., 1.]))
        assert all(mask2.numpy() == np.array([0., 1., 1., 1., 1.]))

        config_list = [{'sparsity': 0.6, 'op_types': ['BatchNorm2d']}]
        model.bn1.weight.data = torch.tensor(w).float()
        model.bn2.weight.data = torch.tensor(w).float()
        pruner = torch_compressor.SlimPruner(model, config_list)

        layer1 = torch_compressor.compressor.LayerInfo('bn1', model.bn1)
        mask1 = pruner.calc_mask(layer1, config_list[0])
        layer2 = torch_compressor.compressor.LayerInfo('bn2', model.bn2)
        mask2 = pruner.calc_mask(layer2, config_list[0])
        assert all(mask1.numpy() == np.array([0., 0., 0., 1., 1.]))
        assert all(mask2.numpy() == np.array([0., 0., 0., 1., 1.]))

    def test_torch_QAT_quantizer(self):
        model = TorchModel()
        config_list = [{
            'quant_types': ['weight'],
            'quant_bits': 8,
            'op_types': ['Conv2d', 'Linear']
        }, {
            'quant_types': ['output'],
            'quant_bits': 8,
            'quant_start_step': 0,
            'op_types': ['ReLU']
        }]
        model.relu = torch.nn.ReLU()
        quantizer = torch_compressor.QAT_Quantizer(model, config_list)
        quantizer.compress()
        # test quantize
        # range not including 0
        eps = 1e-7
        weight = torch.tensor([[1, 2], [3, 5]]).float()
        quantize_weight = quantizer.quantize_weight(weight, config_list[0], model.conv2)
        assert math.isclose(model.conv2.scale, 5 / 255, abs_tol=eps)
        assert model.conv2.zero_point == 0
        # range including 0
        weight = torch.tensor([[-1, 2], [3, 5]]).float()
        quantize_weight = quantizer.quantize_weight(weight, config_list[0], model.conv2)
        assert math.isclose(model.conv2.scale, 6 / 255, abs_tol=eps)
        assert model.conv2.zero_point in (42, 43)

        # test ema
        x = torch.tensor([[-0.2, 0], [0.1, 0.2]])
        out = model.relu(x)
        assert math.isclose(model.relu.tracked_min_biased, 0, abs_tol=eps)
        assert math.isclose(model.relu.tracked_max_biased, 0.002, abs_tol=eps)

        quantizer.step()
        x = torch.tensor([[0.2, 0.4], [0.6, 0.8]])
        out = model.relu(x)
        assert math.isclose(model.relu.tracked_min_biased, 0.002, abs_tol=eps)
        assert math.isclose(model.relu.tracked_max_biased, 0.00998, abs_tol=eps)


if __name__ == '__main__':
    main()
