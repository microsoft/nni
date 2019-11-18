from unittest import TestCase, main
import tensorflow as tf
import torch
import torch.nn.functional as F
import nni.compression.torch as torch_compressor

if tf.__version__ >= '2.0':
    import nni.compression.tensorflow as tf_compressor

def get_tf_mnist_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=7, input_shape=[28, 28, 1], activation='relu', padding="SAME"),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding="SAME"),
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

class TorchMnist(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.fc1 = torch.nn.Linear(4 * 4 * 50, 500)
        self.fc2 = torch.nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def tf2(func):
    def test_tf2_func(self):
        if tf.__version__ >= '2.0':
            func()
    return test_tf2_func

class CompressorTestCase(TestCase):
    def test_torch_pruner(self):
        model = TorchMnist()
        configure_list = [{'sparsity': 0.8, 'op_types': ['default']}]
        torch_compressor.LevelPruner(model, configure_list).compress()

    def test_torch_fpgm_pruner(self):
        model = TorchMnist()
        configure_list = [{'sparsity': 0.5, 'op_types': ['Conv2d']}]
        torch_compressor.FPGMPruner(model, configure_list).compress()

    def test_torch_quantizer(self):
        model = TorchMnist()
        configure_list = [{
            'quant_types': ['weight'],
            'quant_bits': {
                'weight': 8,
            },
            'op_types':['Conv2d', 'Linear']
        }]
        torch_compressor.NaiveQuantizer(model, configure_list).compress()

    @tf2
    def test_tf_pruner(self):
        configure_list = [{'sparsity': 0.8, 'op_types': ['default']}]
        tf_compressor.LevelPruner(get_tf_mnist_model(), configure_list).compress()

    @tf2
    def test_tf_quantizer(self):
        tf_compressor.NaiveQuantizer(get_tf_mnist_model(), [{'op_types': ['default']}]).compress()

    @tf2
    def test_tf_fpgm_pruner(self):
        configure_list = [{'sparsity': 0.5, 'op_types': ['Conv2D']}]
        tf_compressor.FPGMPruner(get_tf_mnist_model(), configure_list).compress()


if __name__ == '__main__':
    main()
