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
* **quant_types:**: 字符串列表 要应用的量化类型，当前支持 'weight', 'input', 'output'
* **quant_bits:** int or dict of {str : int} bits length of quantization, key is the quantization type, value is the length, eg. {'weight', 8}, when the type is int, all quantization types share same bits length
* **quant_start_step:** int disable quantization until model are run by certain number of steps, this allows the network to enter a more stable state where activation quantization ranges do not exclude a signiﬁcant fraction of values, default value is 0

### note
batch normalization folding is currently not supported.
***

## DoReFa Quantizer
在 [DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients](https://arxiv.org/abs/1606.06160) 中，作者 Shuchang Zhou 和 Yuxin Wu 提出了 DoReFa 算法在训练时量化权重，激活函数和梯度。

### Usage
要实现 DoReFa Quantizer，在训练代码前加入以下代码。

TensorFlow 代码
```python
from nni.compressors.tensorflow import DoReFaQuantizer
config_list = [{ 'q_bits': 8, 'op_types': 'default' }]
quantizer = DoReFaQuantizer(tf.get_default_graph(), config_list)
quantizer.compress()
```
PyTorch 代码
```python
from nni.compressors.torch import DoReFaQuantizer
config_list = [{ 'q_bits': 8, 'op_types': 'default' }]
quantizer = DoReFaQuantizer(model, config_list)
quantizer.compress()
```

查看示例进一步了解

#### DoReFa Quantizer 的用户配置
* **q_bits:** 指定需要被量化的位数。
