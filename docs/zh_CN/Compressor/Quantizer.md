NNI Compressor 中的 Quantizer
===
## Naive Quantizer

Naive Quantizer 将 Quantizer 权重默认设置为 8 位，可用它来测试量化算法。

### 用法
Tensorflow
```python
nni.compressors.tensorflow.NaiveQuantizer(model_graph).compress()
```
PyTorch
```python
nni.compressors.torch.NaiveQuantizer(model).compress()
```

***

## QAT Quantizer
在 [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf) 中，作者 Benoit Jacob 和 Skirmantas Kligys 提出了一种算法在训练中量化模型。
> 我们提出了一种方法，在训练的前向过程中模拟量化效果。 此方法不影响反向传播，所有权重和偏差都使用了浮点数保存，因此能很容易的进行量化。 然后，前向传播通过实现浮点算法的舍入操作，来在推理引擎中模拟量化的推理。 * 权重在与输入卷积操作前进行量化。 如果在层中使用了批量归一化（参考 [17]），批量归一化参数会被在量化前被“折叠”到权重中。 * 激活操作在推理时会被量化，例如，在激活函数被应用到卷积或全连接层输出之后，或在增加旁路连接，或连接多个层的输出之后（如：ResNet）。 Activations are quantized at points where they would be during inference, e.g. after the activation function is applied to a convolutional or fully connected layer’s output, or after a bypass connection adds or concatenates the outputs of several layers together such as in ResNets.


### 用法
可在训练代码前将模型量化为 8 位。

PyTorch 代码
```python
from nni.compressors.torch import QAT_Quantizer
model = Mnist()

config_list = [{
    'quant_types': ['weight'],
    'quant_bits': {
        'weight': 8,
    }, # 这里可以仅使用 `int`，因为所有 `quan_types` 使用了一样的位长，参考下方 `ReLu6` 配置。
    'op_types':['Conv2d', 'Linear']
}, {
    'quant_types': ['output'],
    'quant_bits': 8,
    'quant_start_step': 7000,
    'op_types':['ReLU6']
}]
quantizer = QAT_Quantizer(model, config_list)
quantizer.compress()
```

查看示例进一步了解

#### QAT Quantizer 的用户配置
* **quant_types:** : list of string

type of quantization you want to apply, currently support 'weight', 'input', 'output'.

* **op_types:** list of string

specify the type of modules that will be quantized. eg. 'Conv2D'

* **op_names:** list of string

specify the name of modules that will be quantized. eg. 'conv1'

* **quant_bits:** int or dict of {str : int}

bits length of quantization, key is the quantization type, value is the length, eg. {'weight': 8}, when the type is int, all quantization types share same bits length.

* **quant_start_step:** int

disable quantization until model are run by certain number of steps, this allows the network to enter a more stable state where activation quantization ranges do not exclude a signiﬁcant fraction of values, default value is 0

### 注意
batch normalization folding is currently not supported.
***

## DoReFa Quantizer
In [DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients](https://arxiv.org/abs/1606.06160), authors Shuchang Zhou and Yuxin Wu provide an algorithm named DoReFa to quantize the weight, activation and gradients with training.

### 用法
To implement DoReFa Quantizer, you can add code below before your training code

PyTorch code
```python
from nni.compressors.torch import DoReFaQuantizer
config_list = [{ 
    'quant_types': ['weight'],
    'quant_bits': 8, 
    'op_types': 'default' 
}]
quantizer = DoReFaQuantizer(model, config_list)
quantizer.compress()
```

You can view example for more information

#### DoReFa Quantizer 的用户配置
* **quant_types:** : list of string

type of quantization you want to apply, currently support 'weight', 'input', 'output'.

* **op_types:** list of string

specify the type of modules that will be quantized. eg. 'Conv2D'

* **op_names:** list of string

specify the name of modules that will be quantized. eg. 'conv1'

* **quant_bits:** int or dict of {str : int}

bits length of quantization, key is the quantization type, value is the length, eg. {'weight': 8}, when the type is int, all quantization types share same bits length.


## BNN Quantizer
In [Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](https://arxiv.org/abs/1602.02830),
> We introduce a method to train Binarized Neural Networks (BNNs) - neural networks with binary weights and activations at run-time. At training-time the binary weights and activations are used for computing the parameters gradients. During the forward pass, BNNs drastically reduce memory size and accesses, and replace most arithmetic operations with bit-wise operations, which is expected to substantially improve power-efficiency.


### Usage

PyTorch code
```python
from nni.compression.torch import BNNQuantizer
model = VGG_Cifar10(num_classes=10)

configure_list = [{
    'quant_types': ['weight'],
    'quant_bits': 1,
    'op_types': ['Conv2d', 'Linear'],
    'op_names': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'classifier.0', 'classifier.3']
}, {
    'quant_types': ['output'],
    'quant_bits': 1,
    'op_types': ['Hardtanh'],
    'op_names': ['features.6', 'features.9', 'features.13', 'features.16', 'features.20', 'classifier.2', 'classifier.5']
}]

quantizer = BNNQuantizer(model, configure_list)
model = quantizer.compress()
```

You can view example [examples/model_compress/BNN_quantizer_cifar10.py](https://github.com/microsoft/nni/tree/master/examples/model_compress/BNN_quantizer_cifar10.py) for more information.

#### User configuration for BNN Quantizer
* **quant_types:** : list of string

type of quantization you want to apply, currently support 'weight', 'input', 'output'.

* **op_types:** list of string

specify the type of modules that will be quantized. eg. 'Conv2D'

* **op_names:** list of string

specify the name of modules that will be quantized. eg. 'conv1'

* **quant_bits:** int or dict of {str : int}

bits length of quantization, key is the quantization type, value is the length, eg. {'weight': 8}, when the type is int, all quantization types share same bits length.

### Experiment
We implemented one of the experiments in [Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](https://arxiv.org/abs/1602.02830), we quantized the **VGGNet** for CIFAR-10 in the paper. Our experiments results are as follows:

| Model  | Accuracy |
| ------ | -------- |
| VGGNet | 86.93%   |


The experiments code can be found at [examples/model_compress/BNN_quantizer_cifar10.py](https://github.com/microsoft/nni/tree/master/examples/model_compress/BNN_quantizer_cifar10.py) 