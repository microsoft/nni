# 支持的量化算法

支持的量化算法列表
* [Naive Quantizer](#naive-quantizer)
* [QAT Quantizer](#qat-quantizer)
* [DoReFa Quantizer](#dorefa-quantizer)
* [BNN Quantizer](#bnn-quantizer)

## Naive Quantizer

Naive Quantizer 将 Quantizer 权重默认设置为 8 位，可用它来测试量化算法。

### 用法
PyTorch
```python 
model = nni.compression.torch.NaiveQuantizer(model).compress()
```

***

## QAT Quantizer
In [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf), authors Benoit Jacob and Skirmantas Kligys provide an algorithm to quantize the model with training.
> 我们提出了一种方法，在训练的前向过程中模拟量化效果。 此方法不影响反向传播，所有权重和偏差都使用了浮点数保存，因此能很容易的进行量化。 然后，前向传播通过实现浮点算法的舍入操作，来在推理引擎中模拟量化的推理。 * 权重在与输入卷积操作前进行量化。 如果在层中使用了批量归一化（参考 [17]），批量归一化参数会被在量化前被“折叠”到权重中。 * 激活操作在推理时会被量化，例如，在激活函数被应用到卷积或全连接层输出之后，或在增加旁路连接，或连接多个层的输出之后（如：ResNet）。 Activations are quantized at points where they would be during inference, e.g. after the activation function is applied to a convolutional or fully connected layer’s output, or after a bypass connection adds or concatenates the outputs of several layers together such as in ResNets.


### 用法
You can quantize your model to 8 bits with the code below before your training code.

PyTorch code
```python
from nni.compression.torch import QAT_Quantizer
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

You can view example for more information

#### QAT Quantizer 的用户配置

common configuration needed by compression algorithms can be found at [Specification of `config_list`](./QuickStart.md).

configuration needed by this algorithm :

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
from nni.compression.torch import DoReFaQuantizer
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

common configuration needed by compression algorithms can be found at [Specification of `config_list`](./QuickStart.md).

configuration needed by this algorithm :

***

## BNN Quantizer

In [Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](https://arxiv.org/abs/1602.02830),
> 引入了一种训练二进制神经网络（BNN）的方法 - 神经网络在运行时使用二进制权重。 在训练时，二进制权重和激活用于计算参数梯度。 在 forward 过程中，BNN 会大大减少内存大小和访问，并将大多数算术运算替换为按位计算，可显著提高能源效率。


### 用法

PyTorch code
```python
from nni.compression.torch import BNNQuantizer
model = VGG_Cifar10(num_classes=10)

configure_list = [{
    'quant_bits': 1,
    'quant_types': ['weight'],
    'op_types': ['Conv2d', 'Linear'],
    'op_names': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'classifier.0', 'classifier.3']
}, {
    'quant_bits': 1,
    'quant_types': ['output'],
    'op_types': ['Hardtanh'],
    'op_names': ['features.6', 'features.9', 'features.13', 'features.16', 'features.20', 'classifier.2', 'classifier.5']
}]

quantizer = BNNQuantizer(model, configure_list)
model = quantizer.compress()
```

You can view example [examples/model_compress/BNN_quantizer_cifar10.py](https://github.com/microsoft/nni/tree/master/examples/model_compress/BNN_quantizer_cifar10.py) for more information.

#### BNN Quantizer 的用户配置

common configuration needed by compression algorithms can be found at [Specification of `config_list`](./QuickStart.md).

configuration needed by this algorithm :

### 实验

We implemented one of the experiments in [Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](https://arxiv.org/abs/1602.02830), we quantized the **VGGNet** for CIFAR-10 in the paper. Our experiments results are as follows:

| 模型     | 精度     |
| ------ | ------ |
| VGGNet | 86.93% |


The experiments code can be found at [examples/model_compress/BNN_quantizer_cifar10.py](https://github.com/microsoft/nni/tree/master/examples/model_compress/BNN_quantizer_cifar10.py) 