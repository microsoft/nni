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
        self.conv2 = torch.nn.Conv2d(5, 10, 5, 1)
        self.fc1 = torch.nn.Linear(4 * 4 * 10, 100)
        self.fc2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
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

k1 = [[1]*3]*3
k2 = [[2]*3]*3
k3 = [[3]*3]*3
k4 = [[4]*3]*3
k5 = [[5]*3]*3

w = [[k1, k2, k3, k4, k5]] * 10

class CompressorTestCase(TestCase):
    def test_torch_quantizer_modules_detection(self):
        # test if modules can be detected
        model = TorchModel()
        config_list = [{
            'quant_types': ['weight'],
            'quant_bits': 8,
            'op_types':['Conv2d', 'Linear']
        }, {
            'quant_types': ['output'],
            'quant_bits': 8,
            'quant_start_step': 0,
            'op_types':['ReLU']
        }]

        model.relu = torch.nn.ReLU()
        quantizer = torch_compressor.QAT_Quantizer(model, config_list)
        quantizer.compress()
        modules_to_compress = quantizer.get_modules_to_compress()
        modules_to_compress_name = [ t[0].name for t in modules_to_compress]
        assert "conv1" in modules_to_compress_name
        assert "conv2" in modules_to_compress_name
        assert "fc1" in modules_to_compress_name
        assert "fc2" in modules_to_compress_name
        assert "relu" in modules_to_compress_name
    
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
            'op_types':['Conv2d', 'Linear']
        }]
        torch_compressor.NaiveQuantizer(model, configure_list).compress()
    
    @tf2
    def test_tf_naive_quantizer(self):
        tf_compressor.NaiveQuantizer(get_tf_model(), [{'op_types': ['default']}]).compress()

    def test_torch_fpgm_pruner(self):
        """
        With filters(kernels) defined as above (k1 - k5), it is obvious that k3 is the Geometric Median
        which minimize the total geometric distance by defination of Geometric Median in this paper:
        Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration,
        https://arxiv.org/pdf/1811.00250.pdf

        So if sparsity is 0.2, the expected masks should mask out all k3, this can be verified through:
        `all(torch.sum(masks, (0, 2, 3)).numpy() == np.array([90., 90., 0., 90., 90.]))`

        If sparsity is 0.6, the expected masks should mask out all k2, k3, k4, this can be verified through:
        `all(torch.sum(masks, (0, 2, 3)).numpy() == np.array([90., 0., 0., 0., 90.]))`
        """

        model = TorchModel()
        config_list = [{'sparsity': 0.2, 'op_types': ['Conv2d']}, {'sparsity': 0.6, 'op_types': ['Conv2d']}]
        pruner = torch_compressor.FPGMPruner(model, config_list)

        model.conv2.weight.data = torch.tensor(w).float()
        layer = torch_compressor.compressor.LayerInfo('conv2', model.conv2)
        masks = pruner.calc_mask(layer, config_list[0])
        assert all(torch.sum(masks, (0, 2, 3)).numpy() == np.array([90., 90., 0., 90., 90.]))

        pruner.update_epoch(1)
        model.conv2.weight.data = torch.tensor(w).float()
        masks = pruner.calc_mask(layer, config_list[1])
        assert all(torch.sum(masks, (0, 2, 3)).numpy() == np.array([90., 0., 0., 0., 90.]))

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
        masks = masks.transpose([2, 3, 0, 1]).transpose([1, 0, 2, 3])

        assert all(masks.sum((0, 2, 3)) == np.array([90., 90., 0., 90., 90.]))

        pruner.update_epoch(1)
        model.layers[2].set_weights([weights[0], weights[1].numpy()])
        masks = pruner.calc_mask(layer, config_list[1]).numpy()
        masks = masks.transpose([2, 3, 0, 1]).transpose([1, 0, 2, 3])

        assert all(masks.sum((0, 2, 3)) == np.array([90., 0., 0., 0., 90.]))


    def test_torch_QAT_quantizer(self):
        model = TorchModel()
        config_list = [{
            'quant_types': ['weight'],
            'quant_bits': 8,
            'op_types':['Conv2d', 'Linear']
        }, {
            'quant_types': ['output'],
            'quant_bits': 8,
            'quant_start_step': 0,
            'op_types':['ReLU']
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
        assert model.conv2.zero_point == 42

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
